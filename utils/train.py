import os, pickle, numpy as np, pandas as pd, torch, sklearn
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from transformers import EarlyStoppingCallback
from utils.load_data import *  # load_data, tokenized_dataset, RE_Dataset, label_to_num

# ===== Metrics (model-agnostic, dynamic num_labels) =====
def micro_f1_wo_no_relation(preds, labels, label_list: List[str], no_rel: str = "no_relation"):
    no_rel_idx = label_list.index(no_rel)
    use_labels = list(range(len(label_list)))
    use_labels.remove(no_rel_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=use_labels) * 100.0

def auprc_all(probs, labels, num_labels: int):
    labels_oh = np.eye(num_labels)[labels]
    score = np.zeros((num_labels,), dtype=np.float32)
    for c in range(num_labels):
        targets_c = labels_oh[:, c]
        preds_c = probs[:, c]
        p, r, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(r, p)
    return float(np.mean(score) * 100.0)

def make_compute_metrics(label_list: List[str], no_rel: str = "no_relation"):
    num_labels = len(label_list)
    def _compute(eval_pred):
        logits = eval_pred.predictions
        probs  = logits if logits.ndim == 2 else logits[0]
        preds  = probs.argmax(-1)
        labels = eval_pred.label_ids
        return {
            "micro f1 score": micro_f1_wo_no_relation(preds, labels, label_list, no_rel),
            "auprc": auprc_all(probs, labels, num_labels),
            "accuracy": accuracy_score(labels, preds),
        }
    return _compute

# ===== Config =====
DEFAULT_LABEL_LIST = [
    'no_relation', 'org:top_members/employees', 'org:members', 'org:product', 'per:title',
    'org:alternate_names', 'per:employee_of', 'org:place_of_headquarters', 'per:product',
    'org:number_of_employees/members', 'per:children', 'per:place_of_residence',
    'per:alternate_names', 'per:other_family', 'per:colleagues', 'per:origin',
    'per:siblings', 'per:spouse', 'org:founded', 'org:political/religious_affiliation',
    'org:member_of', 'per:parents', 'org:dissolved', 'per:schools_attended',
    'per:date_of_death', 'per:date_of_birth', 'per:place_of_birth', 'per:place_of_death',
    'org:founded_by', 'per:religion'
]

@dataclass
class TrainConfig:
    model_name: str = "klue/bert-base"          # BERT / RoBERTa / ELECTRA 모두 OK
    output_dir: str = "./results"
    num_train_epochs: int = 10
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    seed: int = 42
    max_length: int = 256
    fp16: bool = False                         # True로 주면 A100/3090 등에서 mixed precision
    special_tokens: Optional[List[str]] = None # 예: ["[E1]","[/E1]","[E2]","[/E2]"]
    

# ===== Main train function =====
def train_re(
    train_csv: str,
    dev_csv: Optional[str] = None,
    label_list: List[str] = DEFAULT_LABEL_LIST,
    cfg: TrainConfig = TrainConfig(),
    label_map_path: str = 'dict_label_to_num.pkl',
    save_best_to: str = "./best_model",
):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- 1) Tokenizer / Model (순서 중요) ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # [FIX] 마커 특수토큰을 먼저 추가
    added = 0
    if cfg.special_tokens:
        added = tokenizer.add_special_tokens({"additional_special_tokens": cfg.special_tokens})
        if added > 0:
            print(f"[info] added {added} special tokens")

    num_labels = len(label_list)
    model_config = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=model_config)

    # [FIX] 특수토큰 추가했으면 반드시 임베딩 리사이즈
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---- 2) Load data & label mapping ----
    train_df = load_data(train_csv)
    label_map = {label: idx for idx, label in enumerate(label_list)}

    # [FIX] 라벨이 리스트 밖이면 KeyError 방지
    try:
        train_y = [label_map[v] for v in train_df['label'].values]
    except KeyError as e:
        missing = set(train_df['label'].unique()) - set(label_list)
        raise ValueError(f"Found labels not in label_list: {missing}") from e

    # ---- 3) Tokenize (최종 tokenizer로!) ----
    tokenized_train = tokenized_dataset(train_df, tokenizer)
    # [FIX] RoBERTa 호환: token_type_ids 제거(있으면)
    if isinstance(tokenized_train, dict):
        tokenized_train.pop("token_type_ids", None)

    tokenized_train.pop("token_type_ids", None)

    with torch.no_grad():
        emb = model.get_input_embeddings()
        vocab_size = emb.weight.size(0)
        max_id = int(tokenized_train["input_ids"].max().item())
        print(f"[check] vocab_size={vocab_size}, max_input_id={max_id}")

        if max_id >= vocab_size:
            # 디버그: 어떤 토큰들이 범위를 넘는지 확인
            ids = tokenized_train["input_ids"].view(-1)
            bad_ids = ids[ids >= vocab_size].unique().tolist()
            bad_toks = [tokenizer.convert_ids_to_tokens(int(i)) for i in bad_ids]
            print(f"[warn] out-of-vocab ids: {bad_ids}")
            print(f"[warn] out-of-vocab tokens: {bad_toks}")

            # 1) 가장 보수적인 즉시 복구: 임베딩을 입력의 최대 id+1 로 리사이즈
            new_size = max_id + 1
            print(f"[fix] resize embeddings to {new_size}")
            model.resize_token_embeddings(new_size)
            vocab_size = new_size  # 갱신
    RE_train = RE_Dataset(tokenized_train, train_y)

    if dev_csv is not None:
        dev_df = load_data(dev_csv)
        try:
            dev_y = [label_map[v] for v in dev_df['label'].values]
        except KeyError as e:
            missing = set(dev_df['label'].unique()) - set(label_list)
            raise ValueError(f"[dev] labels not in label_list: {missing}") from e

        tokenized_dev = tokenized_dataset(dev_df, tokenizer)
        tokenized_dev.pop("token_type_ids", None)
        # dev에서도 안전검사(선택)
        with torch.no_grad():
            max_id_dev = int(tokenized_dev["input_ids"].max().item())
            if max_id_dev >= vocab_size:
                raise RuntimeError(
                    f"[dev] Input id ({max_id_dev}) >= embedding size ({vocab_size}). "
                    f"Did tokenizer change after tokenizing?"
                )
        RE_dev = RE_Dataset(tokenized_dev, dev_y)
    else:
        RE_dev = None

    # ---- 4) TrainingArguments (HF 4.55 API: eval_strategy 사용) ----
    evaluation_strategy = 'steps' if RE_dev is not None else 'no'
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        save_total_limit=cfg.save_total_limit,
        save_steps=cfg.save_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_steps=cfg.logging_steps,
        eval_strategy=evaluation_strategy,                              # ← 4.55에서는 eval_strategy
        eval_steps=cfg.eval_steps if RE_dev is not None else None,
        load_best_model_at_end=cfg.load_best_model_at_end if RE_dev is not None else False,
        metric_for_best_model="micro f1 score" if RE_dev is not None else None,
        greater_is_better=True,
        seed=cfg.seed,
        fp16=cfg.fp16,
        remove_unused_columns=False,                                    # RE 커스텀 입력 보존용
        dataloader_pin_memory=torch.cuda.is_available(),   
        report_to="none", 
        # CPU면 경고 억제
    )

    # ---- 5) Trainer ----
    compute_metrics = make_compute_metrics(label_list, no_rel="no_relation") if RE_dev is not None else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_train,
        eval_dataset=RE_dev,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
            callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,        # 연속 2회 개선 없으면 스톱
            early_stopping_threshold=0.002,   # 개선 최소폭(예: F1 0.2% 미만이면 개선으로 안 봄)
        )
    ],
    )

    # ---- 6) Train ----
    trainer.train()

    # ---- 7) Save best (or final) model ----
    os.makedirs(save_best_to, exist_ok=True)
    trainer.save_model(save_best_to)
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(save_best_to)

    print(f"Model saved to: {save_best_to}")
    return trainer

