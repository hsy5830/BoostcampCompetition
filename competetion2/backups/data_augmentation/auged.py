import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

# load data
df = pd.read_csv('../dataset/train/train.csv')

# data augmentation
auged_df = pd.DataFrame([], columns = ['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'])

## 1118, 02:40 -> 17~29 까지
for num_label in tqdm(range(17,30)): # no_relation 제외
    ## sentence 변경하기 - DataFrame
    tmp = df.loc[df['label'] == df['label'].unique()[num_label], :]
    # tmp_df = pd.DataFrame([], columns = ['sentence', 'subject_entity', 'object_entity', 'label'])

    label = df['label'].unique()[num_label]

    for i in tqdm(range(len(tmp))):
        sen = tmp.iloc[i,]['sentence']
        sub_entity = eval(tmp.iloc[i,]['subject_entity'])
        obj_entity = eval(tmp.iloc[i,]['object_entity'])
        sub_s = sub_entity['start_idx']
        sub_e = sub_entity['end_idx']
        obj_s = obj_entity['start_idx']
        obj_e = obj_entity['end_idx']

        tmp_sen = ''
        for sub, obj in zip(tmp['subject_entity'], tmp['object_entity']):
            sub = eval(sub); obj = eval(obj)

            # sentence
            if sub_s < obj_s:
                tmp_sen = sen[:sub_s] + sub['word'] + sen[sub_e+1:obj_s] + obj['word'] + sen[obj_e+1:]
            else:
                tmp_sen = sen[:obj_s] + obj['word'] + sen[obj_e+1:sub_s] + sub['word'] + sen[sub_e+1:]

            # subject_entity
            if sub_s < obj_s:
                sub_ent = {'word':sub['word'] , 'start_idx': sub_s, 'end_idx': sub_s + len(sub['word'])-1, 'type':sub['type']}
            else:
                sub_ent = {'word':sub['word'] , 'start_idx': sub_s + (len(obj['word']) - len(obj_entity['word'])), 'end_idx': sub_s + (len(obj['word']) - len(obj_entity['word'])) + len(sub['word'])-1, 'type':sub['type']}

            # object_entity
            if sub_s < obj_s:
                obj_ent = {'word':obj['word'] , 'start_idx': obj_s + (len(sub['word']) - len(sub_entity['word'])), 'end_idx': obj_s + (len(sub['word']) - len(sub_entity['word'])) + len(obj['word'])-1, 'type':obj['type']}
            else:
                obj_ent = {'word':obj['word'] , 'start_idx': obj_s, 'end_idx': obj_s + len(obj['word'])-1, 'type':obj['type']}

            # id, source
            id_ = 0
            source = 's'

            # concat
            tmp_df0 = pd.DataFrame([[id_,tmp_sen, sub_ent, obj_ent, label, source]], columns = ['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'])
            auged_df = pd.concat([auged_df, tmp_df0])

auged_df.to_csv('train_auged_17_29.csv', encoding="utf-8-sig") 