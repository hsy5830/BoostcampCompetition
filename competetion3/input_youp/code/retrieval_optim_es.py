import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import argparse

from rank_bm25 import BM25Okapi
from scipy import sparse

from email.policy import default
from elastic_setting import *


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        emb_type,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        #################### Splited wiki dataset ####################
        wiki = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        
        splited_wiki = []
        for i in range(len(wiki)):
            temp = re.sub(r"\\n", r"\n", wiki[i])
            splited_wiki.extend(temp.split("\n"))
            
        splited_wiki = [i for i in splited_wiki if i not in ["", " "]] #공백 지우기
        self.contexts = splited_wiki
        ##############################################################

        # self.contexts = list(
        #     dict.fromkeys([v["text"] for v in wiki.values()])
        # )  # set 은 매번 순서가 바뀌므로

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        print(len(self.contexts))
        self.ids = list(range(len(self.contexts)))
        self.tokenizer = tokenize_fn
        self.emb_type = emb_type ###

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000, # embedding의 col 제한
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
    
    def get_sparse_embedding(self) -> NoReturn: # for both TF-IDF or BM25
        if self.emb_type == 'tfidf':
            # Pickle을 저장합니다.
            pickle_name = f"sparse_embedding.bin"
            tfidfv_name = f"tfidv.bin"
            emd_path = os.path.join(self.data_path, pickle_name)
            tfidfv_path = os.path.join(self.data_path, tfidfv_name)

            if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                with open(tfidfv_path, "rb") as file:
                    self.tfidfv = pickle.load(file)
                print("Embedding pickle load.")
            else:
                print("Build passage embedding")
                self.p_embedding = self.tfidfv.fit_transform(self.contexts) # embedding matrix
                print(self.p_embedding.shape)
                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(tfidfv_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                print("Embedding pickle saved.")
        else:
            # Pickle을 저장합니다.
            # pickle_name = f"sparse_embedding.bin"
            bm25v_name = f"bm25v.bin"
            # emd_path = os.path.join(self.data_path, pickle_name)
            bm25v_path = os.path.join(self.data_path, bm25v_name)

            # with timer("bm25 building"):
            #     self.bm25v = BM25Okapi(self.contexts, tokenizer=self.tokenizer) # BM25 객체
            
            if os.path.isfile(bm25v_path):
                with open(bm25v_path, "rb") as file:
                    self.bm25v = pickle.load(file)
                print("BM25 Object load.")        
            else: # 없으면 객체 생성
                with timer("bm25 building"):
                    self.bm25v = BM25Okapi(self.contexts, tokenizer=self.tokenizer) # BM25 객체
                with open(bm25v_path, "wb") as file:
                    pickle.dump(self.bm25v, file)
                print("BM25 Object saved.")


    def build_faiss(self, num_clusters=64) -> NoReturn: # for TF-IDF

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray() # csr_mat -> np.array
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")
            
    def retrieve(  # for both TF-IDF or BM25
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        # 문제가 될 수도 있는 부분
        if self.emb_type == 'tfidf':
            assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk) # score / 큰 순서대로 index
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]: # for both TF-IDF or BM25

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if self.emb_type == 'tfidf':
            with timer("transform"):
                query_vec = self.tfidfv.transform([query])
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            with timer("query ex search"):
                result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_score = result.squeeze()[sorted_result].tolist()[:k] # tf-idf 값
            doc_indices = sorted_result.tolist()[:k] # tf_idf 순으로 정렬한 id 값
            print("#####" * 30)
            print(doc_score) #####
            print("#####" * 30)
            return doc_score, doc_indices
        
        else: # bm25
            with timer("transform"):
                tokenized_query = self.tokenizer(query)
            with timer("query ex search"): ## 오래 걸리는 부분
                result = self.bm25v.get_scores(tokenized_query)
            sorted_result = np.argsort(result)[::-1]
            doc_score = result[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]
            return doc_score, doc_indices

    def get_relevant_doc_bulk( # for both TF-IDF or BM25
        self, queries: List, k: Optional[int] = 1 # for TF-IDF
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if self.emb_type == 'tfidf':
            query_vec = self.tfidfv.transform(queries)
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices
        else:
            with timer("transform"): 
                tokenized_queris = [self.tokenizer(query) for query in queries]
            with timer("query ex search"): ### 오래 걸리는 부분
                result = np.array([self.bm25v.get_scores(tokenized_query) for tokenized_query in tqdm(tokenized_queris)])
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices
    
    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1 # for TF-IDF
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1 # for TF-IDF
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1 # for TF-IDF
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()
    

class ElasticRetrieval:    # 221229 추가
    def __init__(self, INDEX_NAME):
        self.es, self.index_name = es_setting(index_name=INDEX_NAME) 

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1 # 60613
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(min(topk, len(docs))):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc_indices[i])
                print(docs[i]['_source']['document_text'])

            return (doc_scores, [doc_indices[i] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            # ################################# for retrieval ensemble #################################
            # # df.to_csv("./retrieval_ensemble/BM25_b6.5_k2.0.csv", encoding="utf-8-sig", index=False)
            # print("-----" * 10 + " 앙상블 시작 " + "-----" * 10)
            # # topk 변수
            # tmp_indices = [x for x in doc_indices if x != []]

            # result = []
            # # k = 5
            # for i in range(len(doc_scores)):
            #     tmp = doc_scores[i]
            #     tmp_idx = tmp_indices[topk * i : topk * (i+1)]
            #     l0 = [(x, y) for x,y in zip(tmp,tmp_idx)]
            #     l = sorted(l0, key = lambda x: x[1])
            #     result.append([n[0] for n in l])
            # result # (# obs) x (topk) list
            # result = pd.DataFrame(result)

            # print("-----" * 20)
            # print(result.shape)
            # print("-----" * 20)

            # result.to_csv("./retrieval_ensemble/BM25_b6.5_k2.0.csv", encoding="utf-8-sig", index=False)

            # ##########################################################################################
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")):
                # retrieved_context 구하는 부분 수정
                retrieved_context = []
                for i in range(min(topk, len(docs[idx]))):
                    retrieved_context.append(docs[idx][i]['_source']['document_text'])

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"], # question id
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    # "context_id": doc_indices[idx],
                    "context": " ".join(retrieved_context),  # 수정
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            # cqas.to_csv("./retrieval_ensemble/wowwow.csv", encoding="utf-8-sig", index=False)
            return cqas
    
    
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        doc_score = []
        doc_index = []
        res = es_search(self.es, self.index_name, query, k)
        docs = res['hits']['hits']
        # print("######################################## 가보자고 ########################################")
        # print("######################################## 가보자고 ########################################")
        # print("######################################## 가보자고 ########################################")
        # pd.DataFrame(docs).to_csv("./retrieval_ensemble/corpus_id.csv", encoding="utf-8-sig", index=False)

        for hit in docs:
            doc_score.append(hit['_score'])
            doc_index.append(hit['_id'])
            print("Doc ID: %3r  Score: %5.2f" % (hit['_id'], hit['_score']))

        return doc_score, doc_index, docs
    
    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        total_docs = []
        doc_scores = []
        doc_indices = []

        for query in queries:
            doc_score = []
            doc_index = []
            res = es_search(self.es, self.index_name, query, k)
            docs = res['hits']['hits']
        

            for hit in docs:
                doc_score.append(hit['_score'])
                doc_indices.append(hit['_id'])

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)
            total_docs.append(docs)
        # print("######################################## 가보자고 ########################################")
        # print("######################################## 가보자고 ########################################")
        # print("######################################## 가보자고 ########################################")
        # pd.DataFrame(total_docs).to_csv("./retrieval_ensemble/corpus_id.csv", encoding="utf-8-sig", index=False)
        # ################################# for retrieval ensemble #################################
        # print("-----" * 10 + " 앙상블 시작 " + "-----" * 10)
        # # topk 변수
        # tmp_indices = [x for x in doc_indices if x != []]

        # result = []
        # # k = 5
        # for i in range(len(doc_scores)):
        #     tmp = doc_scores[i]
        #     tmp_idx = tmp_indices[k * i : k * (i+1)]
        #     tmp.append(tmp_idx) # DataFrame에서 score1, ..., scorek, [indices] 로 표현하기 위함
        #     result.append(tmp)
        # # result # (# obs) x (topk) list
        # result = pd.DataFrame(result)

        # print("-----" * 20)
        # print(result.shape)
        # print("-----" * 20)

        # # result.to_csv("./retrieval_ensemble/BM25_b0.55_k1.8.csv", encoding="utf-8-sig", index=False)
        # # result.to_csv("./retrieval_ensemble/DFI_standardized.csv", encoding="utf-8-sig", index=False)
        # # result.to_csv("./retrieval_ensemble/IB_ll_ttf_h3.csv", encoding="utf-8-sig", index=False)
        # # result.to_csv("./retrieval_ensemble/LMJelinekMercer_0.8.csv", encoding="utf-8-sig", index=False)
        # ##########################################################################################

        return doc_scores, doc_indices, total_docs


def main(args):
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    test_dataset = load_from_disk("../data/test_true/test")
    # test_dataset = load_from_disk("../data/test_true_splited")
    full_ds = concatenate_datasets(
        [
            # org_dataset["train"].flatten_indices(),
            # org_dataset["validation"].flatten_indices(),
            test_dataset.flatten_indices(), # test로만 accuracy 계산
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    print("*" * 93)
    
    if args.elastic:
        retriever = ElasticRetrieval(args.index_name)
        print("Using ElasticRetrieval")
    else:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)
        print("#" * 100)
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
            emb_type = args.emb_type,
        )
        print("#" * 100)
        ## 초기에만 실행
        retriever.get_sparse_embedding()
        retriever.build_faiss()
        ##

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        # # 기존 우진 수정버전
        # with timer("single query by exhaustive search"):
        #     scores, indices = retriever.retrieve(query)

        # with timer("bulk query by exhaustive search"):
        #     df = retriever.retrieve(full_ds, topk = args.topk)
        #     correct = []
            
        #     for i in range(len(df)):
        #         if df['original_context'][i] in df['context'][i]:
        #             correct.append(1)
        #         else:
        #             correct.append(0)

        #     # df["correct"] = df["original_context"] == df["context"]
        #     df['correct'] = correct
        #     print(
        #         # "correct retrieval result by exhaustive search",
        #         f"top-{args.topk} correct retrieval result by exhaustive search",
        #         df["correct"].sum() / len(df),
        #     )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
        
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.topk)
            df["correct"] = [original_context in context for original_context,context in zip(df["original_context"],df["context"])]
            print(
                "correct retrieval result by exhaustive search",
                f"{df['correct'].sum()}/{len(df)}",
                df["correct"].sum() / len(df),
            )
            accuracy = df["correct"].sum() / len(df)
            return accuracy



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    ## retrieval_optim_es.py
    parser.add_argument(
        "--dataset_name", metavar="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="../data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, default=False, help="")
    
    parser.add_argument("--emb_type", metavar='bm25', type=str, help="")
    parser.add_argument("--topk", metavar = 3, type = int, help = "")

    parser.add_argument("--elastic", default=True, type=bool, help="Elasticsearch를 사용하지 않는 경우 False로 설정해주세요")
    parser.add_argument("--index_name", default="origin-wiki", type=str, help="테스트할 index name을 설정해주세요")

    ## elastic_setting.py
    parser.add_argument("--setting_path", default="./setting.json", type=str, help="생성할 index의 setting.json 경로를 설정해주세요")
    parser.add_argument("--dataset_path", default="../data/wikipedia_documents.json", type=str, help="삽입할 데이터의 경로를 설정해주세요")
    # parser.add_argument("--index_name", default="origin-wiki", type=str, help="테스트할 index name을 설정해주세요")

    ## optim_es.py
    parser.add_argument("--types", metavar='bm25', type=str, help="")
    # bm25
    
    args = parser.parse_args()
    main(args)