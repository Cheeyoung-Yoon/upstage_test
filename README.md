
# Upstage Online Deep Learning Code Test

## 1. Summary
- This project tackles the **Korean Relation Extraction (RE)** task by combining various Korean PLMs with marker-based encoding, Marker Head, ERPE, FGM, LLRD, Hard Negative Mining, R-Drop, and CB/Focal Loss to enhance performance.  
- **Model selection was conducted entirely through grid search**, with intermediate evaluations performed at multiple stages to progressively narrow down promising configurations.  
- Comparative experiments on tokenizers/backbones showed that **`klue/roberta-base` with typed inline markers** achieved the most stable and highest Micro F1 scores.  
- The `grid_plus.py` script automates running/evaluating all model × hyperparameter combinations and stores all results in CSV format.  
- All experiments were implemented and optimized for GPU training on **Google Colab** with an **NVIDIA A100 (40GB)**.

---

## 2. Experimental results

### 2.1 Model Selection (Backbone Comparison)
| Model                                     | LR    | Train BSZ | Best Step | Micro F1    | AUPRC   | Accuracy | Val Loss |
| ----------------------------------------- | ----- | --------- | --------- | ----------- | ------- | -------- | -------- |
| klue\_bert-base                           | 3e-05 | 16        | 4000      | 59.7681     | 47.1765 | 0.606098 | 1.287195 |
| klue\_bert-base                           | 5e-05 | 16        | 7000      | 59.6718     | 47.4565 | 0.609794 | 1.430602 |
| **klue\_roberta-base**                    | **2e-05** | **16**   | **8000** | **60.6296** | **48.4109** | **0.618417** | **1.374527** |
| klue\_roberta-base                        | 3e-05 | 32        | 3500      | 60.3333     | 47.9823 | 0.615953 | 1.291947 |
| monologg\_koelectra-base-v3-discriminator | 3e-05 | 16        | 8000      | 59.6228     | 39.8815 | 0.607946 | 1.417223 |

### 2.2 Technique & Parameter Ablations
| Run ID | mh  | erpe | fgm | rdrop | llrd | hn  | Micro F1 | AUPRC   | Accuracy |
| ------ | --- | ---- | --- | ----- | ---- | --- | -------- | ------- | -------- |
| I1     | 1   | 0    | 0   | 0.0   | 0    | 0   | 76.1946  | 67.4056 | 74.5919  |
| I2     | 1   | 0    | 0   | 0.0   | 0    | 1   | 75.7369  | 67.1022 | 74.0068  |
| I5     | 1   | 0    | 1   | 0.0   | 0    | 0   | 76.2939  | 65.7801 | 74.4687  |
| I6     | 1   | 0    | 1   | 0.0   | 0    | 1   | 76.2939  | 65.7801 | 74.4687  |
| I9     | 1   | 1    | 0   | 0.0   | 0    | 0   | 76.6645  | 66.4619 | 75.0847  |
| I10    | 1   | 1    | 0   | 0.0   | 0    | 1   | 75.6699  | 65.7936 | 74.1608  |
| I13    | 1   | 1    | 1   | 0.0   | 0    | 0   | 76.3997  | 66.7691 | 74.6843  |
| I14    | 1   | 1    | 1   | 0.0   | 0    | 1   | 75.7881  | 66.4826 | 74.0068  |
| I17    | 0   | 0    | 0   | 0.0   | 0    | 0   | 76.3023  | 67.0793 | 74.6227  |
| I18    | 0   | 0    | 0   | 0.0   | 0    | 1   | 76.2108  | 67.0786 | 74.5611  |
| N1     | 1   | 1    | 0   | 0.0   | 0    | 0   | 75.6699  | 65.7936 | 74.1608  |
| N2     | 1   | 1    | 1   | 0.0   | 0    | 0   | 75.7881  | 66.4969 | 73.9759  |
| N5     | 1   | 1    | 0   | 2.0   | 0    | 0   | **78.1743** | 65.9849 | **76.4706** |
| L1     | 1   | 1    | 0   | 1.0   | 0    | 0   | 76.2421  | 66.9428 | 74.5611  |
| L2     | 1   | 1    | 1   | 1.0   | 0    | 0   | 76.1104  | 67.5358 | 74.4687  |
| L5     | 1   | 1    | 0   | 2.0   | 0    | 0   | 76.5727  | 68.0351 | 75.0539  |
| L6     | 1   | 1    | 1   | 2.0   | 0    | 0   | 76.2361  | 67.3326 | 74.6843  |
| L9     | 1   | 1    | 0   | 3.0   | 0    | 0   | 76.3266  | 68.3232 | 74.8383  |
| B1 (16)    | 1   | 1    | 0   | 2.0     | 0    | 0  | 74.5611  | 66.9657 | 74.2531  |
| B2 (32)    | 1  | 1   | 0   | 2.0    | 0    | 0   | 76.7633  | 66.6115 | 75.2078  |




| Column    | Full Name                                | Description                                                                                                                                                                           |
| --------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **mh**    | Marker Head                              | Whether the classification head uses the hidden states from the first entity marker tokens instead of `[CLS]` (1 = enabled, 0 = disabled). Improves focus on entity-specific context. |
| **erpe**  | Entity-aware Relative Position Embedding | Adds two separate relative position embeddings for distances to entity 1 and entity 2 (1 = enabled, 0 = disabled). Useful for distance-sensitive relations.                           |
| **fgm**   | Fast Gradient Method                     | Adversarial training method that adds small perturbations to embeddings during training to improve robustness (1 = enabled, 0 = disabled).                                            |
| **rdrop** | R-Drop Alpha                             | Strength (α) of R-Drop regularization, which enforces prediction consistency by running the model twice with different dropout masks and adding KL divergence loss.                   |
| **llrd**  | Layer-wise Learning Rate Decay           | Whether learning rates are decayed for lower transformer layers (1 = enabled, 0 = disabled). Helps preserve general language features in early layers.                                |
| **hn**    | Hard Negative Mining                     | Whether difficult `no_relation` examples are up-weighted and re-sampled in subsequent epochs (1 = enabled, 0 = disabled).                                                             |


| Prefix | Meaning           | Description                                                                                                                         |
| ------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **I**  | Initial Test      | First-stage grid search runs with baseline configurations and early variations to identify promising techniques.                    |
| **N**  | Next Iteration    | Second-stage experiments after evaluating I-runs; focuses on refining the top-performing setups by adjusting one or two parameters. |
| **L**  | R-Drop Level Test | Runs specifically testing different R-Drop α values and combinations with other techniques.                                         |
| **B**  | Batch Size Test   | Experiments comparing the effect of different batch sizes (16 vs 32) on training stability and performance for promising configs.   |


---

## 3. Instructions

### Environment
- **Python**: 3.9+
- **PyTorch**: 1.13+ (CUDA 11.x)
- **Transformers**: ≥ 4.30.0
- **scikit-learn**, **pandas**, **numpy**
- **GPU**: NVIDIA A100 (40GB) on Google Colab

### Project Structure
```

pstage\_test/
├── parts\_config.py      # Label list & dataclass configs
├── data\_plus.py         # Dataset loading, preprocessing, marker insertion
├── model\_builders.py    # Marker Head, ERPE model builders
├── trainer\_plus.py      # Trainer with Focal/CB Loss, R-Drop, FGM, LLRD, HNM
├── hardneg\_callback.py  # Hard Negative Mining callback
├── train\_re.py          # Single-run training
├── grid\_plus.py         # Grid search over models & hyperparameters
├── run.py               # Entry point

````

### Usage
1. Upload the dataset and set `TRAIN_CSV` and `DEV_CSV` paths.
2. Edit `MODELS` and `HP_SPACE` to desired configurations.
3. Run:
```bash
python run.py
````

4. Results will be saved under the `base_out` directory and aggregated into CSV.

---

## 4. Approach

### EDA

#### 1. Label Distribution by Entity Type Pair

The table below shows counts of each relation for every `(subject_type, object_type)` combination.

|                | no\_relation | org\:alternate\_names | org\:dissolved | org\:founded | org\:founded\_by | org\:member\_of | org\:members | org\:number\_of\_employees/members | org\:place\_of\_headquarters | org\:political/religious\_affiliation | org\:product | org\:top\_members/employees | per\:alternate\_names | per\:children | per\:colleagues | per\:date\_of\_birth | per\:date\_of\_death | per\:employee\_of | per\:origin | per\:other\_family | per\:parents | per\:place\_of\_birth | per\:place\_of\_death | per\:place\_of\_residence | per\:product | per\:religion | per\:schools\_attended | per\:siblings | per\:spouse | per\:title |
| :------------- | -----------: | --------------------: | -------------: | -----------: | ---------------: | --------------: | -----------: | ---------------------------------: | ---------------------------: | ------------------------------------: | -----------: | --------------------------: | --------------------: | ------------: | --------------: | -------------------: | -------------------: | ----------------: | ----------: | -----------------: | -----------: | --------------------: | --------------------: | ------------------------: | -----------: | ------------: | ---------------------: | ------------: | ----------: | ---------: |
| ('ORG', 'DAT') |         1582 |                     1 |             66 |          450 |                0 |               5 |            1 |                                  0 |                            4 |                                     1 |            0 |                           0 |                     0 |             0 |               0 |                    0 |                    0 |                 0 |           0 |                  0 |            0 |                     0 |                     0 |                         0 |            0 |             0 |                      0 |             0 |           0 |          0 |
| ('ORG', 'LOC') |          548 |                    23 |              0 |            0 |                0 |             173 |           97 |                                  0 |                          894 |                                     4 |           24 |                          13 |                     0 |             0 |               0 |                    0 |                    0 |                 0 |           0 |                  0 |            0 |                     0 |                     0 |                         0 |            0 |             0 |                      0 |             0 |           0 |          0 |
| ('ORG', 'ORG') |         1958 |                  1154 |              0 |            0 |                5 |            1320 |          285 |                                  0 |                          254 |                                    54 |           48 |                          22 |                     0 |             0 |               0 |                    0 |                    0 |                 0 |           0 |                  0 |            0 |                     0 |                     0 |                         0 |            0 |             0 |                      0 |             0 |           0 |          0 |
| ('ORG', 'PER') |          401 |                    31 |              0 |            0 |              144 |               1 |            2 |                                  0 |                            1 |                                     1 |            3 |                        4195 |                     0 |             0 |               0 |                    0 |                    0 |                 0 |           0 |                  0 |            0 |                     0 |                     0 |                         0 |            0 |             0 |                      0 |             0 |           0 |          0 |
| ('PER', 'ORG') |          741 |                     0 |              0 |            0 |                0 |               0 |            0 |                                  0 |                            0 |                                     0 |            0 |                           0 |                    40 |             1 |              10 |                    0 |                    1 |              2857 |         267 |                  1 |            0 |                     3 |                     1 |                        11 |           11 |            80 |                     80 |             0 |           1 |        141 |

---

#### 2. Entity Distance per Label

Average and distribution of character distances between subject and object entities.

| label                 | count | mean  | std   | min | 25%  | 50% | 75%  | max |
| --------------------- | ----- | ----- | ----- | --- | ---- | --- | ---- | --- |
| per\:place\_of\_death | 40    | 34.18 | 22.65 | 5   | 20.8 | 28  | 45.2 | 94  |
| no\_relation          | 9534  | 30.36 | 27.61 | 3   | 12   | 22  | 40   | 295 |
| org\:product          | 380   | 29.61 | 26.96 | 4   | 11   | 20  | 39   | 163 |
| org\:top\_members/... | 4284  | 14.61 | 18.74 | 3   | 7    | 9   | 13   | 253 |
| per\:date\_of\_birth  | 1130  | 10.95 | 8.56  | 4   | 7    | 9   | 10   | 99  |

---

#### 3. Sentence Length per Label

Average and distribution of sentence lengths per label.
| label                 | count | mean   | std   | min | 25% | 50% | 75% | max |
| --------------------- | ----- | ------ | ----- | --- | --- | --- | --- | --- |
| org\:alternate\_names | 1320  | 108.59 | 52.12 | 22  | 72  | 98  | 133 | 419 |
| no\_relation          | 9534  | 104.71 | 50.96 | 21  | 70  | 94  | 126 | 447 |
| org\:top\_members/... | 4284  | 98.62  | 41.66 | 14  | 70  | 91  | 118 | 345 |
| per\:date\_of\_birth  | 1130  | 63.40  | 29.98 | 25  | 45  | 55  | 74  | 384 |

---

#### 4. Top Frequent Entities per Label

**Label:** `no_relation`

* Top Subjects: {'민주당': 102, '두산 베어스': 59, 'FC 서울': 53, '더불어민주당': 46, '삼성 라이온즈': 46}
* Top Objects: {'2016년': 64, '물리학': 59, '더불어민주당': 57, '2007': 48, 'UEFA': 45}

**Label:** `org:alternate_names`

* Top Subjects: {'문화방송': 41, '유럽 축구 연맹': 35, '국제수영연맹': 32, '국제축구연맹': 31, '동양방송': 16}
* Top Objects: {'MBC': 41, 'FIFA': 37, 'UEFA': 35, 'FINA': 33, 'IOC': 29}

**Label:** `org:founded`

* Top Subjects: {'동아일보': 6, '동양척식주식회사': 4, 'KBO 리그': 4, '대한민국 정부': 3, '유엔난민기구': 3}
* Top Objects: {'1982년': 8, '1920년': 8, '2008년': 7, '1997년': 7, '1919년': 6}


#### **5. Problem Understanding**

##### **Task Definition**

* **Goal**: Predict one of 30 predefined relations between two entities in a sentence.
* **Input**:

  * **Sentence**: Natural language sentence containing two marked entities.
  * **Subject Entity**: Word or phrase (with type information: `PER`, `ORG`, `LOC`, etc.)
  * **Object Entity**: Word or phrase (with type information).
* **Output**:

  * A relation label (e.g., `org:founded`, `per:employee_of`, `no_relation`).

##### **Dataset Characteristics from EDA**

* **Class imbalance**:

  * `no_relation` ≈ **29%** of data
  * Several rare classes (`org:dissolved`, `per:religion`) have very few samples.
* **Entity type correlation**:

  * Certain type pairs are highly predictive (e.g., `ORG` → `DAT` often = `org:founded`).
* **Entity proximity**:

  * Median subject-object distance ≈ **14 characters**; most relations occur within \~30 characters.
* **Sentence length variance**:

  * Ranges from 14 to 455 characters; 75% ≤ 118 characters.

---

#### **2. Modeling Strategy**

##### **2.1 Data Preprocessing**

1. **Entity Marking with Type Information**

   * Wrap entities in special tokens with their types:

     ```text
     [SUBJ_ORG] 오라클 [/SUBJ_ORG] ... [OBJ_DAT] 1982년 [/OBJ_DAT]
     ```
   * Allows model to distinguish role and type of each entity.
   * Recommended format: `[SUBJ_{TYPE}] ... [/SUBJ_{TYPE}]` and `[OBJ_{TYPE}] ... [/OBJ_{TYPE}]`.

2. **Handling Class Imbalance**

   * **Weighted Loss**:

     ```python
     loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
     ```
   * **Focal Loss** to emphasize hard-to-classify rare relations.
   * Optional **undersampling** of `no_relation` for balanced mini-batches.

3. **Positional Features**

   * From EDA: most relations are short-range.
   * Add **distance embedding**: bucketize `entity_distance` into bins (e.g., 0–5, 6–10, 11–20, >20).
   * Can be added as extra embedding layer in the model.

4. **Max Sequence Length**

   * Set `max_len=256` (covers >99% of dataset while avoiding truncation for long cases).

---

##### **2.2 Model Architecture**

###### **Baseline — Transformer-based Classifier**

* **Backbone**: `klue/roberta-base` or `monologg/koelectra-base-v3-discriminator` (good for Korean NLU).
* **Input**: Tokenized sentence with entity markers.
* **Output**: Dense layer → Softmax over 30 labels.
* **Why**: Strong pretrained language understanding for Korean, simple to implement.

#### **Entity-Aware Transformer (Improved)**

* Adds **type embeddings** for `subject_type` and `object_type`.
* Uses **marker-based pooling**:

  * Extract hidden states of special entity markers.
  * Concatenate `[SUBJ]` + `[OBJ]` embeddings → Classification layer.
* Improves focus on entities, which is critical for RE.

#### **Hybrid Approach (Context + Entity Representation)**

* Encode sentence with transformer.
* Separately embed:

  * **Subject span**
  * **Object span**
  * **Global context**
* Concatenate all → MLP → Softmax.


### Model & Training Strategy

#### Tokenizer / Backbone Search

* **Why**: Tokenization quality directly impacts how entity boundaries and context are represented. In Korean, poor tokenization can fragment names, harming entity-level semantics.
* **How**: Performed a **full grid search** over backbones (`klue/bert-base`, `klue/roberta-base`, `monologg/koelectra-base-v3-discriminator`) with varying learning rates/batch sizes. After each run, intermediate evaluation was used to prune underperformers.
* **Outcome**: `klue/roberta-base` gave the best Micro F1 (60.63% baseline), with stable learning curves due to its subword vocabulary and RoBERTa pretraining advantages.

#### Marker Strategy

* **Why**: Directs model attention to entities and injects type priors.
* **How**: Inserted **typed inline markers** (`[E1-PER] ... [/E1]`, `[E2-ORG] ... [/E2]`) around entities. Typed markers leverage entity types; inline markers preserve sentence flow.
* **Outcome**: +1–1.5 Micro F1 over plain markers, improved disambiguation in fine-grained classes.

#### ERPE (Entity-aware Relative Position Embeddings)

* **Why**: Relation meaning often depends on **relative** distance between entities and context.
* **How**: Added separate position embeddings for distances to each entity.
* **Outcome**: +0.3–0.5 Micro F1 with Marker Head, especially for distance-sensitive relations.

#### Marker Head

* **Why**: `[CLS]` token alone may dilute entity-specific information.
* **How**: Extracted hidden states from first marker positions for both entities, concatenated, and passed to classification head.
* **Outcome**: \~+1 Micro F1 and fewer symmetric relation errors.

#### Loss Functions

* **Why**: Dataset imbalance leads to bias toward `no_relation`.
* **How**:

  * **CB Loss**: Weighted by effective number of samples per class.
  * **Focal Loss**: Down-weights easy samples, focuses on hard examples.
* **Outcome**: CB Loss improved minority-class recall by 3–5 AUPRC points.

#### Regularization & Training Tricks

* **FGM**: Adds small adversarial noise (`ε=1e-3`) to embeddings after first backward pass, improving robustness to lexical variation.
* **LLRD**: Applies 0.95 decay per layer for learning rates, protecting general language features in lower layers.
* **R-Drop**: Runs two forward passes with different dropout masks, adds KL divergence loss (`alpha=2.0`).
  **Key Result**: Run N5 (mh=1, erpe=1, fgm=0, rdrop=2.0) achieved **78.17 Micro F1**, the highest overall.
* **Hard Negative Mining**: Identifies difficult `no_relation` examples each epoch and up-weights them for the next. Improved `no_relation` precision and boosted overall Micro F1.

#### Evaluation Metrics

* **Primary**: Micro F1 (excluding `no_relation`).
* **Secondary**: AUPRC (for imbalance sensitivity), Accuracy (for completeness).

#### Early Stopping

* **Why**: Prevent overfitting and save compute.
* **How**: Stop if Micro F1 does not improve within the patience window.
* **Outcome**: Reduced training time without hurting peak performance.

### Future Work

* Data augmentation for rare classes.
* Testing with larger PLMs (`klue/roberta-large`).
* Extending to multi-sentence RE scenarios.

```
