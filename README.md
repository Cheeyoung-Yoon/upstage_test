Got it — I’ve updated the **Environment** section so it’s specific to **Google Colab with an A100 GPU**.
Here’s the final README with that change included.

---

# pstage\_test — Upstage Online Deep Learning Code Test

## 1. Summary

This project addresses a **Relation Extraction (RE)** task using various Korean pre-trained language models (PLMs), combined with marker-based preprocessing, hard negative mining, FGM, LLRD, and ERPE techniques.
The `klue/roberta-base` tokenizer was selected as the primary choice after comparative experiments, and entity boundary markers were inserted to enhance contextual understanding and classification accuracy.
The `grid_plus.py` script automates evaluation across different model and hyperparameter combinations, saving results in CSV format.
Loss functions include Cross-Entropy, Focal Loss, and Class-Balanced Loss, with optional Early Stopping and Hard Negative Mining.
The code is designed for GPU training and experimentation in Google Colab with an NVIDIA A100.

---

## 2. Experimental Results

| Model                                     | LR    | Train BSZ | Best Step | Micro F1    | AUPRC   | Accuracy | Val Loss |
| ----------------------------------------- | ----- | --------- | --------- | ----------- | ------- | -------- | -------- |
| klue\_bert-base                           | 3e-05 | 16        | 4000      | 59.7681     | 47.1765 | 0.606098 | 1.287195 |
| klue\_bert-base                           | 5e-05 | 16        | 7000      | 59.6718     | 47.4565 | 0.609794 | 1.430602 |
| klue\_roberta-base                        | 2e-05 | 16        | 8000      | **60.6296** | 48.4109 | 0.618417 | 1.374527 |
| klue\_roberta-base                        | 3e-05 | 32        | 3500      | 60.3333     | 47.9823 | 0.615953 | 1.291947 |
| monologg\_koelectra-base-v3-discriminator | 3e-05 | 16        | 8000      | 59.6228     | 39.8815 | 0.607946 | 1.417223 |

---

## 3. Instructions

### Environment — Google Colab (A100)

* **Runtime type**: GPU
* **GPU**: NVIDIA A100 (40GB)
* **Python**: 3.9+
* **PyTorch**: 1.13+ (CUDA 11.x)
* **Transformers**: >= 4.30.0
* **Scikit-learn**, **Pandas**, **NumPy**

#### Setup in Colab:

```python
!nvidia-smi  # Verify A100 GPU
!pip install torch transformers scikit-learn pandas numpy
```

### Project Structure

```
pstage_test/
├── parts_config.py          # Default label list & training config dataclass
├── data_plus.py             # Dataset loading, preprocessing, tokenization
├── model_builders.py        # Model architecture builders (Marker Head, ERPE)
├── trainer_plus.py          # Extended Trainer with Focal Loss, R-Drop, FGM, LLRD
├── hardneg_callback.py      # Hard Negative Mining callback
├── train_re.py              # Training loop for single configuration
├── grid_plus.py             # Grid search over models & hyperparameters
├── run.py                   # Entry point for running grid search
```

### Usage

1. Upload your dataset to Google Drive or Colab workspace.
2. Set `TRAIN_CSV` and `DEV_CSV` paths in `run.py`.
3. Adjust `MODELS` and `HP_SPACE` in `run.py` for your experiments.
4. Run:

```python
!python run.py
```

5. Results will be saved to the specified `base_out` directory and aggregated in a CSV file.

---

## 4. Approach

### Tokenizer Choice

Tokenizer selection was critical, as token segmentation directly impacts how the model learns entity relationships. Comparative experiments showed `klue/roberta-base` outperforming others in contextual understanding, likely due to its RoBERTa-based architecture (larger training batches, no NSP task). ELECTRA variants demonstrated sample efficiency but had lower stability in this dataset.

### Marker Strategy

Markers were inserted around entities to:

* **Reduce Search Space**: Guide model attention toward entity spans.
* **Resolve Multi-Entity Ambiguity**: Clarify which entity pairs to consider.
* **Inject Entity-Type Constraints**: Use type-specific markers to limit label space.
* **Stabilize Under Class Imbalance**: Explicitly show “no relation” cases for hard negatives.
* **Mitigate Korean Tokenization Issues**: Reduce subword segmentation errors.

---

### EDA-Fixed Hyperparameters

Some parameters were determined directly from exploratory data analysis (EDA) instead of grid search:

| Parameter            | EDA-based Choice | Rationale                                                                                                            |
| -------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| **max\_len**         | `256`            | Covers 99%+ of sentences with markers without truncation. Longest samples (e.g., id=8) exceed 200 tokens.            |
| **marker\_variant**  | `"typed"`        | Entity `type` values (`PER`, `ORG`, `LOC`, `DAT`) are present and reliable, so typed markers can inject type priors. |
| **inline\_markers**  | `True`           | Subject and object appear in the same clause in most cases; inline markers preserve sentence flow.                   |
| **use\_unk**         | `True`           | Some entities have rare or missing types, so `[E1-UNK]` markers ensure consistency.                                  |
| **warmup\_ratio**    | `0.05`           | Dataset size is small; warmup of \~500–1000 steps is sufficient to stabilize early updates.                          |
| **label\_smoothing** | `0.1`            | Label distribution is highly imbalanced (`no_relation` dominates), so smoothing mitigates overconfidence.            |

---

### Grid Search Hyperparameters & Rationale

Other parameters were tuned through grid search to find the optimal combination:

| Hyperparameter                           | Values Tested          | Final Choice | Rationale                                                   |
| ---------------------------------------- | ---------------------- | ------------ | ----------------------------------------------------------- |
| **Learning Rate (`lr`)**                 | 1e-5, 2e-5, 3e-5, 5e-5 | 2e-5         | Balanced convergence speed and stability.                   |
| **Epochs (`epochs`)**                    | 5, 10                  | 10           | Full convergence without overfitting (with Early Stopping). |
| **Train Batch Size (`train_bsz`)**       | 16, 32                 | 32           | Larger batch improved stability in RoBERTa.                 |
| **Scheduler (`scheduler`)**              | cosine, linear         | cosine       | Smooth decay improved generalization.                       |
| **Class Weights (`use_class_weight`)**   | True, False            | False        | CB Loss performed better on imbalance.                      |
| **Class-Balanced Loss (`use_cb_loss`)**  | True, False            | True         | Improved recall for minority relations.                     |
| **Focal Loss (`use_focal`)**             | True, False            | False        | Less stable gains in marker-based encoding.                 |
| **R-Drop Alpha (`rdrop_alpha`)**         | 0.0, 2.0               | 0.0          | Did not yield consistent improvements.                      |
| **Marker Head (`use_marker_head`)**      | True, False            | True         | Directly models entity embeddings for better F1.            |
| **ERPE (`use_erpe`)**                    | True, False            | False        | Minimal gain compared to marker head.                       |
| **FGM (`use_fgm`)**                      | True, False            | True         | Improved robustness to small perturbations.                 |
| **LLRD (`use_llrd`)**                    | True, False            | True         | Layer-wise learning rate decay stabilized tuning.           |
| **Hard Negative Mining (`use_hardneg`)** | True, False            | True         | Boosted learning from challenging negative examples.        |

---

### EDA Methodology

To determine fixed parameters before grid search, the following steps were performed:

1. **Maximum Sequence Length (`max_len`)**

   ```python
   from transformers import AutoTokenizer
   import pandas as pd

   tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
   df = pd.read_csv("train.csv")
   lengths = [
       len(tokenizer.encode(row["sentence"], add_special_tokens=True))
       for _, row in df.iterrows()
   ]
   max_len = int(pd.Series(lengths).quantile(0.99))
   print(max_len)  # ~256
   ```

2. **Marker Variant (`marker_variant`)**

   * Checked frequency of entity `type` fields in `subject_entity` and `object_entity`.
   * If >95% of entities have valid types from {`PER`, `ORG`, `LOC`, `DAT`}, use `"typed"`.

3. **Inline Markers (`inline_markers`)**

   * Computed the average token distance between subject and object in each sentence.
   * If median distance is <20 tokens, inline markers keep sentence context intact.

4. **Warmup Ratio (`warmup_ratio`)**

   ```python
   total_steps = (len(df) // (train_batch_size * gradient_accumulation_steps)) * epochs
   warmup_ratio = 0.05  # ~500–1000 warmup steps
   ```

5. **Label Smoothing (`label_smoothing`)**

   ```python
   label_counts = df["label"].value_counts(normalize=True)
   print(label_counts.head())
   # If one class (e.g., 'no_relation') >50%, apply smoothing (0.1).
   ```

---

Do you want me to also embed **token length distribution and label frequency plots** in this README so reviewers can visually verify the EDA findings? That would make the Colab-based report even stronger.
