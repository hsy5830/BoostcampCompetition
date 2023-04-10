from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference(model, tokenized_sent, device,  pair_type):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=64, shuffle=False) #이건 배치사이즈 상관없나???
  #16, 64로 바꿔가며 실험해보았더니, 완전 미세하게 다르다 확률값이 0.00001단위로 다르다...?
  # 걍 아무거나 해도 될듯?
  

  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    
    ## 여기서 prob 수정하기(현재 pair_type별 후보 레이블 개수만큼 나오는데 30개로 만들기!)
    with open(f'recent_pickle/dict_infer_pair_type_{pair_type}.pkl', 'rb') as f:
      dict2 = pickle.load(f)

    real_prob = [[0]*30 for _ in range(len(prob))] #batch_size * 30
    
    for i in range(len(prob)): #sample_prob은 배치에서 하나의 샘플의 확률이다(소프트맥스 통과 값)
      for j in range(len(prob[i])):
        real_prob[i][dict2[j]]=prob[i][j]


    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result) ##여기서는 0,1,2,3, 이런 값이지만 뒤에 main에서 num_to_label로 올바르게 바꿔준다! 걱정 ㄴㄴ
    output_prob.append(real_prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label, pair_type):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  num_label_real = []
  with open(f'recent_pickle/dict_infer_pair_type_{pair_type}.pkl', 'rb') as f:
    dict2 = pickle.load(f)

  for v in label:
    num_label_real.append(dict2[v])
 
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in num_label_real:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label ## 문자열로 된 label의 리스트

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)


  # test_label = list(map(int,test_dataset['label'].values)) ##test의 label은 문자열 "100" 이다. -> 숫자 100으로 바꿔준다. 따라서 숫자100의 리스트
  test_label = [100]*len(test_dataset['label'].values)##test의 label은 문자열 "100" 이다. -> 숫자 100으로 바꿔준다. 따라서 숫자100의 리스트
  
  
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = 'klue/roberta-base'
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)



  ## load my model 
  best_list = [0,0,0,0,0,0,306,196,21,536,790,270]
  for pair_type in range(6,12):


    # MODEL_NAME = f"./best_model/{'-'.join(Tokenizer_NAME.split('/'))}/recent/type_{pair_type}/batch16"
    MODEL_NAME = f"./results/{'-'.join(Tokenizer_NAME.split('/'))}/recent/type_{pair_type}/batch64/checkpoint-{best_list[pair_type]}"

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = f"../dataset/test/recent/test_pair_type_{pair_type}.csv"
    #test_dataset_dir = f"../dataset/train/recent/dev0_pair_type_{pair_type}.csv"

    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label)

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device, pair_type) # model에서 class 추론
    pred_answer = num_to_label(pred_answer, pair_type) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

    output.to_csv(f'./prediction/recent/submission_type_{pair_type}.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

  ##최종 concat한 결과 만들기

  temp = []
  for pair_type in range(6,12):
    temp2 = pd.read_csv(f"./prediction/recent/submission_type_{pair_type}.csv")
    temp.append(temp2)


  submission_final = pd.concat(temp, ignore_index=True)
  submission_final = submission_final.sort_values(by = ['id'], ascending = True)

  ## 하나 빠진거 집어 넣기(6820번째 데이터가 빠지게 됨)
  temp = submission_final.loc[0,]
  temp["id"]=6820
  temp = pd.DataFrame(temp).T

  submission_final = pd.concat([submission_final, temp], ignore_index=True, axis = 0)
  submission_final = submission_final.sort_values(by = ['id'], ascending = True)

  submission_final.to_csv(f'./prediction/recent/submission_final.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  

  

  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # # model dir
  # parser.add_argument('--model_dir', type=str, default="./best_model")

  ## 실행할떄 python3 inference_recent.py --model_dir=f'./results/{"-".join(args.MODEL_NAME.split("/"))}/recent/type_{args.pair_type}/batch{w_config.batch_size}'
  # parser.add_argument('--pair_type', default=0, type=int)
  ##<내 주석>
  ## 현재 model_dir의 default 값은 "./best_model" 이다.
  ## 만약 results 폴더에 저장된 checkpoint 사용하려고 하면,
  ## ai_stage에서 말해준것 처럼(https://stages.ai/competitions/207/data/requirements)
  ## ex> python inference.py --model_dir=./results/checkpoint-500 이런식으로 사용해야 한다!

  args = parser.parse_args()
  print(args)
  main(args)
  
