import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset


def tokenized_dataset(dataset, tokenizer, use_type_markers=True, use_unk=True, max_len=256):
    """
    dataset: pandas.DataFrame with columns:
      - sentence
      - subject_entity, object_entity  (dict-like str: {'word':..., 'type':...})
    """
    import ast

    def parse_ent(e):
        if isinstance(e, str):
            try:
                e = ast.literal_eval(e)
            except:
                return None, None
        if isinstance(e, dict):
            return e.get("word"), e.get("type")
        return None, None

    enc_inputs, enc_texts = [], []

    for s_ent, o_ent, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
        s_word, s_type = parse_ent(s_ent)
        o_word, o_type = parse_ent(o_ent)

        # 단어가 누락된 경우 안전장치
        s_word = s_word if s_word else "<SUBJ>"
        o_word = o_word if o_word else "<OBJ>"

        if use_type_markers:
            if not s_type and use_unk: s_type = "UNK"
            if not o_type and use_unk: o_type = "UNK"

            if s_type and o_type:
                e_span = f"[E1-{s_type}]{s_word}[/E1] [E2-{o_type}]{o_word}[/E2]"
            else:
                # 타입을 전혀 모르면 타입 없는 일반 마커 사용
                e_span = f"[E1]{s_word}[/E1] [E2]{o_word}[/E2]"
        else:
            # 타입 마커 비활성화: 일반 마커만
            e_span = f"[E1]{s_word}[/E1] [E2]{o_word}[/E2]"

        enc_inputs.append(e_span)
        enc_texts.append(sent)

    # 필요 시 특수 토큰 등록 (한 번만 실행)
    # 타입 마커/일반 마커/종료 마커 + UNK
    special_tokens = {"additional_special_tokens": [
        "[E1]","[/E1]","[E2]","[/E2]",
        "[E1-PER]","[E2-PER]","[E1-ORG]","[E2-ORG]",
        "[E1-LOC]","[E2-LOC]","[E1-UNK]","[E2-UNK]"
    ]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    # model.resize_token_embeddings(len(tokenizer))  # 모델 로드 후 1회 실행

    return tokenizer(
        enc_inputs,
        enc_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )

