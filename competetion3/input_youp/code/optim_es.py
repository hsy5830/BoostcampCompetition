import elastic_setting
import retrieval_optim_es

import json
import pprint
import warnings
import re
import argparse
from tqdm import tqdm
import pandas as pd
from elasticsearch import Elasticsearch

topk_l = [1, 10, 20, 40]
types = ['BM25', "DFI", "LMDirichlet", "LMJelinekMercer", "IB", "DFR"]
# types = ["LMJelinekMercer", "IB", "DFR"] # tfidf, bm25 완료
types = ['BM25']

# BM25
# b = [0.55, 0.65, 0.75, 0.85, 0.95]
# k1 = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
b = [0.65, 0.75]
k1 = [1.6, 1.8, 2.0]

# DFR
# basic_model = ['g', 'if', 'in', 'ine']
# after_effect = ['b', 'l']
# normalization = ['no', 'h1', 'h2', 'h3', 'z']
basic_model = ['g']
after_effect = ['b']
normalization_ = ['h2']

# DFI
independence_measure = ["standardized", "saturated", "chisquared"]


# IB
# distribution = ["ll", "spl"]
# lambda_ = ["df", "ttf"]
# normalization = ['no', 'h1', 'h2', 'h3', 'z']
distribution = ["ll"]
lambda_ = ["ttf"]
normalization = ['h3']

# LM Dirichlet
mu = [800, 900, 1000]
mu = [800]

# LM Jelinek Mercer
lambda__ = [0.6, 0.65, 0.7, 0.8, 0.9]
lambda__ = [0.8]


def main(args):
    results = []
    ### setiing.json 불러와서 파일 수정하기
    with open(args.setting_path, "r") as f:
        setting = json.load(f)
    
    for num_types in range(len(types)):
        args.types = types[num_types]
        # settings -> analysis -> similarity -> my_similarity -> type, b, k1
        if args.types == 'BM25':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            ## uni, bi, tri 설정
            # "min_shingle_size": 2,
            # "max_shingle_size": 4                            
            for i in range(len(b)):
                for j in range(len(k1)):
                    s = {
                        "settings": {
                            "analysis": {
                                "filter": {
                                    "my_shingle": {
                                        "type": "shingle"
                                    }
                                },
                                "analyzer": {
                                    "my_analyzer": {
                                        "type": "custom",
                                        "tokenizer": "nori_tokenizer",
                                        "decompound_mode": "mixed",
                                        "filter": ["my_shingle"]
                                    }
                                }
                            },
                            "index": {
                                "similarity": {
                                    "my_similarity": {
                                        "type": "BM25",
                                        "b": b[i],
                                        "k1": k1[j]
                                    }
                                }
                            }
                        },
                        "mappings": {
                            "properties": {
                                "document_text": {
                                    "type": "text",
                                    "analyzer": "my_analyzer",
                                    "similarity": "my_similarity"
                                }
                            }
                        }
                    }
                    s = str(s)
                    s = re.sub("'", '"', s)
                    with open('./setting.json', 'w+') as lf:
                        lf.write(s)
                    ### elastic_setting.main(args) 실행하기
                    # embedding matrix 만드는 과정
                    elastic_setting.main(args)
                    
                    # top 1, 10, 20, 40
                    p = re.compile(r'"my_similarity": \{[a-zA-Z0-9\s":,.]+\}')
                    m = p.search(s)
                    for k in range(len(topk_l)):
                        args.topk = topk_l[k]
                        acc = retrieval_optim_es.main(args)
                        results.append([acc, args.topk, m.group()])
                        r.append([acc, args.topk, m.group()])
            r = list(map(str, r))
            with open('./results_BM25_splited.txt', 'w+') as lf:
                lf.write('\n'.join(r))            
            
        elif args.types == 'TFIDF':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            s = {
                    "settings": {
                        "analysis": {
                            "filter": {
                                "my_shingle": {
                                    "type": "shingle"
                                }
                            },
                            "analyzer": {
                                "my_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "decompound_mode": "mixed",
                                    "filter": ["my_shingle"]
                                }
                            }
                        },
                        "index": {
                            "similarity": {
                                "my_similarity": {
                                    "type": "scripted",
                                    "script": {
                                            "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;"
                                        }
                                    }
                                }
                            }
                        },
                    "mappings": {
                        "properties": {
                            "document_text": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity"
                            }
                        }
                    }
                    }
            s = str(s)
            s = re.sub("'", '"', s)
            with open('./setting.json', 'w+') as lf:
                lf.write(s)
            ### elastic_setting.main(args) 실행하기
            # embedding matrix 만드는 과정
            elastic_setting.main(args)
            
            # top 1, 10, 20, 40
            p = re.compile(r'"my_similarity": \{[a-zA-Z0-9\s":,.]+\}')
            m = p.search(s) 
            for i in range(len(topk_l)):
                args.topk = topk_l[i]
                acc = retrieval_optim_es.main(args)
                results.append(["tf-idf", args.topk ,acc])
                r.append([acc, args.topk, m.group()])
            
            r = list(map(str, r))
            with open('./results_TFIDF.txt', 'w+') as lf:
                lf.write('\n'.join(r))
        
        elif args.types == 'DFR':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            for i in range(len(basic_model)):
                for j in range(len(after_effect)):
                    for k in range(len(normalization_)):
                        s = {
                            "settings": {
                                "analysis": {
                                    "filter": {
                                        "my_shingle": {
                                            "type": "shingle"
                                        }
                                    },
                                    "analyzer": {
                                        "my_analyzer": {
                                            "type": "custom",
                                            "tokenizer": "nori_tokenizer",
                                            "decompound_mode": "mixed",
                                            "filter": ["my_shingle"]
                                        }
                                    }
                                },
                                "index": {
                                    "similarity": {
                                        "my_similarity": {
                                            "type": "DFR",
                                            "basic_model": basic_model[i],
                                            "after_effect": after_effect[j],
                                            "normalization": normalization_[k],
                                        }
                                    }
                                }
                            },
                            "mappings": {
                                "properties": {
                                    "document_text": {
                                        "type": "text",
                                        "analyzer": "my_analyzer",
                                        "similarity": "my_similarity"
                                    }
                                }
                            }
                        }
                        s = str(s)
                        s = re.sub("'", '"', s)
                        with open('./setting.json', 'w+') as lf:
                            lf.write(s)

                        ### elastic_setting.main(args) 실행하기
                        # embedding matrix 만드는 과정      
                        elastic_setting.main(args)

                        # top 1, 10, 20, 40
                        p = re.compile(r'"my_similarity": \{[\[\]a-zA-Z0-9\s":,._]+\}')
                        m = p.search(s)
                        for k in range(len(topk_l)):
                            args.topk = topk_l[k]
                            acc = retrieval_optim_es.main(args)
                            results.append([acc, args.topk, m.group()])
                            r.append([acc, args.topk, m.group()])
            r = list(map(str, r))
            with open('./results_DFR.txt', 'w+') as lf:
                lf.write('\n'.join(r))

        elif args.types == 'DFI':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            for i in range(len(independence_measure)):
                s = {
                    "settings": {
                        "analysis": {
                            "filter": {
                                "my_shingle": {
                                    "type": "shingle"
                                }
                            },
                            "analyzer": {
                                "my_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "decompound_mode": "mixed",
                                    "filter": ["my_shingle"]
                                }
                            }
                        },
                        "index": {
                            "similarity": {
                                "my_similarity": {
                                    "type": "DFI",
                                    "independence_measure": independence_measure[i]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "document_text": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity"
                            }
                        }
                    }
                }
                s = str(s)
                s = re.sub("'", '"', s)
                with open('./setting.json', 'w+') as lf:
                    lf.write(s)

                ### elastic_setting.main(args) 실행하기
                # embedding matrix 만드는 과정      
                elastic_setting.main(args)

                # top 1, 10, 20, 40
                p = re.compile(r'"my_similarity": \{[a-zA-Z0-9\s":,.\[\]_]+\}')
                m = p.search(s)
                for k in range(len(topk_l)):
                    args.topk = topk_l[k]
                    acc = retrieval_optim_es.main(args)
                    results.append([acc, args.topk, m.group()])
                    r.append([acc, args.topk, m.group()])
            r = list(map(str, r))
            with open('./results_DFI.txt', 'w+') as lf:
                lf.write('\n'.join(r))

        elif args.types == 'IB':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            for i in range(len(distribution)):
                for j in range(len(lambda_)):
                    for k in range(len(normalization)):
                        s = {
                            "settings": {
                                "analysis": {
                                    "filter": {
                                        "my_shingle": {
                                            "type": "shingle"
                                        }
                                    },
                                    "analyzer": {
                                        "my_analyzer": {
                                            "type": "custom",
                                            "tokenizer": "nori_tokenizer",
                                            "decompound_mode": "mixed",
                                            "filter": ["my_shingle"]
                                        }
                                    }
                                },
                                "index": {
                                    "similarity": {
                                        "my_similarity": {
                                            "type": "IB",
                                            "distribution": distribution[i],
                                            "lambda": lambda_[j],
                                            "normalization": normalization[k],
                                        }
                                    }
                                }
                            },
                            "mappings": {
                                "properties": {
                                    "document_text": {
                                        "type": "text",
                                        "analyzer": "my_analyzer",
                                        "similarity": "my_similarity"
                                    }
                                }
                            }
                        }
                        s = str(s)
                        s = re.sub("'", '"', s)
                        with open('./setting.json', 'w+') as lf:
                            lf.write(s)
                        
                        ### elastic_setting.main(args) 실행하기
                        # embedding matrix 만드는 과정      
                        elastic_setting.main(args)

                        # top 1, 10, 20, 40
                        p = re.compile(r'"my_similarity": \{[a-zA-Z0-9\s":,.]+\}')
                        m = p.search(s)     
                        for k in range(len(topk_l)):
                            args.topk = topk_l[k]
                            acc = retrieval_optim_es.main(args)
                            results.append([acc, args.topk, m.group()])
                            r.append([acc, args.topk, m.group()])
            r = list(map(str, r))
            with open('./results_IB.txt', 'w+') as lf:
                lf.write('\n'.join(r))

        elif args.types == 'LMDirichlet':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            for i in range(len(mu)):
                s = {
                    "settings": {
                        "analysis": {
                            "filter": {
                                "my_shingle": {
                                    "type": "shingle"
                                }
                            },
                            "analyzer": {
                                "my_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "decompound_mode": "mixed",
                                    "filter": ["my_shingle"]
                                }
                            }
                        },
                        "index": {
                            "similarity": {
                                "my_similarity": {
                                    "type": "LMDirichlet",
                                    "mu": mu[i]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "document_text": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity"
                            }
                        }
                    }
                }
                s = str(s)
                s = re.sub("'", '"', s)
                with open('./setting.json', 'w+') as lf:
                    lf.write(s)
                
                ### elastic_setting.main(args) 실행하기
                # embedding matrix 만드는 과정      
                elastic_setting.main(args)

                # top 1, 10, 20, 40
                p = re.compile(r'"my_similarity": \{[a-zA-Z0-9\s":,.\[\]]+\}')
                m = p.search(s)
                for k in range(len(topk_l)):
                    args.topk = topk_l[k]
                    acc = retrieval_optim_es.main(args)
                    results.append([acc, args.topk, m.group()])
                    r.append([acc, args.topk, m.group()])
            r = list(map(str, r))
            with open('./results_LMDirichlet.txt', 'w+') as lf:
                lf.write('\n'.join(r))

        elif args.types == 'LMJelinekMercer':
            print("##########" * 10)
            print(args.types)
            print("##########" * 10)
            r = []
            for i in range(len(lambda__)):
                s = {
                    "settings": {
                        "analysis": {
                            "filter": {
                                "my_shingle": {
                                    "type": "shingle"
                                }
                            },
                            "analyzer": {
                                "my_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "nori_tokenizer",
                                    "decompound_mode": "mixed",
                                    "filter": ["my_shingle"]
                                }
                            }
                        },
                        "index": {
                            "similarity": {
                                "my_similarity": {
                                    "type": "LMJelinekMercer",
                                    "lambda": lambda__[i]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "document_text": {
                                "type": "text",
                                "analyzer": "my_analyzer",
                                "similarity": "my_similarity"
                            }
                        }
                    }
                }
                s = str(s)
                s = re.sub("'", '"', s)
                with open('./setting.json', 'w+') as lf:
                    lf.write(s)

                ### elastic_setting.main(args) 실행하기
                # embedding matrix 만드는 과정      
                elastic_setting.main(args)

                # top 1, 10, 20, 40
                p = re.compile(r'"my_similarity": \{[a-zA-Z0-9\s":,._\[\]]+\}')
                m = p.search(s)
                for k in range(len(topk_l)):
                    args.topk = topk_l[k]
                    acc = retrieval_optim_es.main(args)
                    results.append([acc, args.topk, m.group()])
                    r.append([acc, args.topk, m.group()])
            r = list(map(str, r))
            with open('./results_LMJelinekMercer.txt', 'w+') as lf:
                lf.write('\n'.join(r))


    results = list(map(str, results))
    with open('./results.txt', 'w+') as lf:
        lf.write('\n'.join(results))


if __name__ == "__main__":
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
    print(args)
    print(type(args))
    main(args)