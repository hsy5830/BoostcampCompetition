import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *


## wandb & sweep 추가
import wandb

import argparse

import os
import random
import numpy as np

## seed 추가
def seed_everything(seed: int = 42, contain_cuda: bool = False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set as {seed}")


##이부분 수정함(binary)
def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1"""
    label_list = ['no_relation', "else"]
    label_indices = list(range(len(label_list)))
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'accuracy': acc,
  }

##이부분 수정함(binary)
def label_to_num(label):
  num_label = []
  for v in label:
    if v =="no_relation":
      num_label.append(0)
    else:
      num_label.append(1)  
  return num_label

# def train():
#   # load model and tokenizer
#   # MODEL_NAME = "bert-base-uncased"
#   # MODEL_NAME = "klue/bert-base"
#   # MODEL_NAME = "klue/roberta-base"
#   MODEL_NAME = "klue/roberta-large"
#   # MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
#   # MODEL_NAME = "monologg/kobert"
#   # MODEL_NAME = "kykim/t5-kor-small" #Tokenizer 어쩌구저쩌구 문제 뜸
#   # MODEL_NAME = "kykim/gpt3-kor-small_based_on_gpt2" #TypeError 뜸
#   # MODEL_NAME = "kykim/funnel-kor-base"
#   # MODEL_NAME = "kykim/electra-kor-base"
#   # MODEL_NAME = ""
#   # MODEL_NAME = ""

  





#   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#   # load dataset
#   train_dataset = load_data("../dataset/train/train.csv")
#   # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

#   train_label = label_to_num(train_dataset['label'].values)
#   # dev_label = label_to_num(dev_dataset['label'].values)

#   # tokenizing dataset
#   tokenized_train = tokenized_dataset(train_dataset, tokenizer)
#   # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

#   # make dataset for pytorch.
#   RE_train_dataset = RE_Dataset(tokenized_train, train_label)
#   # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#   print(device)
#   # setting model hyperparameter
#   model_config =  AutoConfig.from_pretrained(MODEL_NAME)
#   model_config.num_labels = 30 ##디폴트 값으로 2로 설정 되어 있다

#   model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
#   ##<내 주석>
#   ##만약 불러올때 conifg를 직접 우리가 입력하는게 아니라면 MODEL_NAME으로
#   ##허깅페이스 타고 들어가서 거기에 있는 model_config 파일을 자동으로 가져온다.
#   ##만약 우리가 직접 입력해주면 우리가 입력해준걸 사용한다!
  
#   print(model.config)
#   model.parameters
#   model.to(device)


  
#   # 사용한 option 외에도 다양한 option들이 있습니다.
#   # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
#   training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     save_total_limit=5,              # number of total save model.
#     save_steps=500,                 # model saving step.
#     num_train_epochs=1,              # total number of training epochs
#     learning_rate=5e-5,               # learning_rate
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=16,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=100,              # log saving step.
#     evaluation_strategy='steps', # evaluation strategy to adopt during training
#                                 # `no`: No evaluation during training.
#                                 # `steps`: Evaluate every `eval_steps`.
#                                 # `epoch`: Evaluate every end of epoch.
#     eval_steps = 500,            # evaluation step.
#     load_best_model_at_end = True ##이게 중요한 역할을 해주는 것 같다! ## 찾아보기!!!
#   )
#   trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=RE_train_dataset,         # training dataset
#     eval_dataset=RE_train_dataset,             # evaluation dataset
#     compute_metrics=compute_metrics         # define metrics function
#   )

#   # train model
#   trainer.train()
#   model.save_pretrained('./best_model')



# def main():
#   train()


if __name__ == '__main__':


  # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
  # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
  # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', default=42, type=int)
  parser.add_argument('--MODEL_NAME', default='klue/roberta-base', type=str)
  ## 대문자임에 주의!!!(MODEL_NAME)

  # args = parser.parse_args(args=[]) ##이거때문에 안되었던 것
  args = parser.parse_args() #이렇게 하니까 된다

  seed_everything(args.seed)

  # method
  sweep_config = {
      'method': 'grid' # random, grid, bayes
  }


  # hyperparameters
  parameters_dict = {
      # 'epochs': {
      #     'value': [1,2]
      #     },
      'batch_size': {
          'values': [64]
          },
      # 'learning_rate': {
      #     'distribution': 'log_uniform_values',
      #     'min': 1e-5,
      #     'max': 1e-3
      # },
      # 'weight_decay': {
      #     'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
      # },
  }

  sweep_config['parameters'] = parameters_dict

  sweep_id = wandb.sweep(
        sweep_config, # config 딕셔너리를 추가합니다.
        project='angyungjabi_binary', # project의 이름을 추가합니다.
        entity="bc_nlp_fp")


  def sweep_train(config=None):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    # MODEL_NAME = "klue/bert-base"
    # MODEL_NAME = "klue/roberta-base"
    # MODEL_NAME = "klue/roberta-large"
    # MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
    # MODEL_NAME = "monologg/kobert"
    # MODEL_NAME = "kykim/t5-kor-small" #Tokenizer 어쩌구저쩌구 문제 뜸
    # MODEL_NAME = "kykim/gpt3-kor-small_based_on_gpt2" #TypeError 뜸
    # MODEL_NAME = "kykim/funnel-kor-base"
    # MODEL_NAME = "kykim/electra-kor-base"
    # MODEL_NAME = ""
    # MODEL_NAME = ""

    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME)

    # load dataset
    train_dataset = load_data("../dataset/train/train_binary.csv")
    dev_dataset = load_data("../dataset/train/dev_binary.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(args.MODEL_NAME)
    
    ##이부분 수정함(binary)
    model_config.num_labels = 2 ##디폴트 값으로 2로 설정 되어 있다

    model =  AutoModelForSequenceClassification.from_pretrained(args.MODEL_NAME, config=model_config)
    ##<내 주석>
    ##만약 불러올때 conifg를 직접 우리가 입력하는게 아니라면 MODEL_NAME으로
    ##허깅페이스 타고 들어가서 거기에 있는 model_config 파일을 자동으로 가져온다.
    ##만약 우리가 직접 입력해주면 우리가 입력해준걸 사용한다!
    
    print(model.config)
    model.parameters
    model.to(device)

    #####여기부터 수정
    wandb.init(config=config)

    # set sweep configuration
    w_config = wandb.config

    ## wandb에 저장되는 이름 지정
    wandb.run.name = f'{"-".join(args.MODEL_NAME.split("/"))}_binary_batch{w_config.batch_size}' ##이거 이름에 / 들어가 있어서 불안하긴 한데...
    

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
      # output_dir='./results',          # output directory
      output_dir = f'./results/{"-".join(args.MODEL_NAME.split("/"))}/binary/batch{w_config.batch_size}',

      ##새롭게 추가
      report_to='wandb',  # Turn on Weights & Biases logging
      # run_name=f"model_{w_config.batch_size}",  # name of the W&B run (optional)
      
      save_total_limit=5,              # number of total save model.
      save_steps=500,                 # model saving step.
      num_train_epochs=10,              # total number of training epochs
      learning_rate=5e-5,               # learning_rate
      per_device_train_batch_size= w_config.batch_size,  # batch size per device during training
      per_device_eval_batch_size=w_config.batch_size,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps = 500,            # evaluation step.
      load_best_model_at_end = True ##이게 중요한 역할을 해주는 것 같다! ## 찾아보기!!!
    )
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      eval_dataset=RE_dev_dataset,             # evaluation dataset
      compute_metrics=compute_metrics         # define metrics function
    )

    print(args)
    print("woojin jjjjang")

    # train model
    trainer.train()
    
    #model.save_pretrained('./best_model')
    model.save_pretrained(f'./best_model/{"-".join(args.MODEL_NAME.split("/"))}/binary/batch{w_config.batch_size}')




  wandb.agent(
        sweep_id,      # sweep의 정보를 입력하고
        function=sweep_train,   # train이라는 모델을 학습하는 코드를
        count=10               # 총 10회 실행해봅니다.(grid의 경우 10번이 안되어도 다하면 멈추는 듯)
    )