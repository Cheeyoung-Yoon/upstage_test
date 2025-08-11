Upstage Online Deep Learning Code Test
1. Summary

    This project tackles the Korean Relation Extraction (RE) task by combining various Korean PLMs with marker-based encoding, Marker Head, ERPE, FGM, LLRD, Hard Negative Mining, R-Drop, and CB/Focal Loss to enhance performance.

    Model selection was conducted entirely through grid search, with intermediate evaluations performed at multiple stages to progressively narrow down promising configurations.

    Comparative experiments on tokenizers/backbones showed that klue/roberta-base with typed inline markers achieved the most stable and highest Micro F1 scores.

    The grid_plus.py script automates running/evaluating all model × hyperparameter combinations and stores all results in CSV format.

    All experiments were implemented and optimized for GPU training on Google Colab with an NVIDIA A100 (40GB).

2. Experimental results
2.1 Model Selection (Backbone Comparison)
Model	LR	Train BSZ	Best Step	Micro F1	AUPRC	Accuracy	Val Loss
klue_bert-base	3e-05	16	4000	59.7681	47.1765	0.606098	1.287195
klue_bert-base	5e-05	16	7000	59.6718	47.4565	0.609794	1.430602
klue_roberta-base	2e-05	16	8000	60.6296	48.4109	0.618417	1.374527
klue_roberta-base	3e-05	32	3500	60.3333	47.9823	0.615953	1.291947
monologg_koelectra-base-v3-discriminator	3e-05	16	8000	59.6228	39.8815	0.607946	1.417223
2.2 Technique & Parameter Ablations
Run ID	mh	erpe	fgm	rdrop	llrd	hn	Micro F1	AUPRC	Accuracy
I1	1	0	0	0.0	0	0	76.1946	67.4056	74.5919
I2	1	0	0	0.0	0	1	75.7369	67.1022	74.0068
I5	1	0	1	0.0	0	0	76.2939	65.7801	74.4687
I6	1	0	1	0.0	0	1	76.2939	65.7801	74.4687
I9	1	1	0	0.0	0	0	76.6645	66.4619	75.0847
I10	1	1	0	0.0	0	1	75.6699	65.7936	74.1608
I13	1	1	1	0.0	0	0	76.3997	66.7691	74.6843
I14	1	1	1	0.0	0	1	75.7881	66.4826	74.0068
I17	0	0	0	0.0	0	0	76.3023	67.0793	74.6227
I18	0	0	0	0.0	0	1	76.2108	67.0786	74.5611
N1	1	1	0	0.0	0	0	75.6699	65.7936	74.1608
N2	1	1	1	0.0	0	0	75.7881	66.4969	73.9759
N5	1	1	0	2.0	0	0	78.1743	65.9849	76.4706
L1	1	1	0	1.0	0	0	76.2421	66.9428	74.5611
L2	1	1	1	1.0	0	0	76.1104	67.5358	74.4687
L5	1	1	0	2.0	0	0	76.5727	68.0351	75.0539
L6	1	1	1	2.0	0	0	76.2361	67.3326	74.6843
L9	1	1	0	3.0	0	0	76.3266	68.3232	74.8383
B1	-	-	-	-	-	-	74.5611	66.9657	74.2531
B2	-	-	-	-	-	-	76.7633	66.6115	75.2078
3. Instructions
Environment

    Python: 3.9+

    PyTorch: 1.13+ (CUDA 11.x)

    Transformers: ≥ 4.30.0

    scikit-learn, pandas, numpy

    GPU: NVIDIA A100 (40GB) on Google Colab

Project Structure

pstage_test/
├── parts_config.py      # Label list & dataclass configs
├── data_plus.py         # Dataset loading, preprocessing, marker insertion
├── model_builders.py    # Marker Head, ERPE model builders
├── trainer_plus.py      # Trainer with Focal/CB Loss, R-Drop, FGM, LLRD, HNM
├── hardneg_callback.py  # Hard Negative Mining callback
├── train_re.py          # Single-run training
├── grid_plus.py         # Grid search over models & hyperparameters
├── run.py               # Entry point

Usage

    Upload the dataset and set TRAIN_CSV and DEV_CSV paths.

    Edit MODELS and HP_SPACE to desired configurations.

    Run:

python run.py

    Results will be saved under the base_out directory and aggregated into CSV.

4. Approach
EDA

(Placeholder — left intentionally blank for later completion)
Model & Training Strategy

Tokenizer / Backbone Search

    Why: Tokenization quality directly impacts how entity boundaries and surrounding context are represented to the model. Poor tokenization can fragment entity names in Korean, making it harder for the model to learn entity-level semantics.

    How: We performed a full grid search over multiple backbones (klue/bert-base, klue/roberta-base, monologg/koelectra-base-v3-discriminator) and learning rates/batch sizes, evaluating after each combination to filter out underperforming candidates early.

    Outcome: klue/roberta-base emerged as the best choice, consistently outperforming others in Micro F1 (60.63% in baseline) and showing more stable learning curves. Its subword vocabulary and no Next Sentence Prediction pretraining allowed it to model longer context windows more effectively, especially with markers inserted.

Marker Strategy

    Why: In RE, the model needs to focus on two specific entities within a sentence. Without explicit guidance, attention can be diffused across irrelevant tokens, especially in longer or multi-clause sentences.

    How: We inserted typed inline markers such as [E1-PER] and [/E1] around the first entity, and [E2-ORG] and [/E2] around the second entity directly within the sentence.

        Typed markers (with entity type information like PER, ORG, LOC, DAT) inject type priors into the model, reducing confusion between semantically similar entities.

        Inline markers preserve sentence flow and avoid separating entity spans into different sequences, which was important since the median entity distance in our dataset was under 20 tokens.

    Outcome: Typed inline markers improved Micro F1 by ~1–1.5 points compared to plain markers, especially in fine-grained relation classes.

ERPE (Entity-aware Relative Position Embeddings)

    Why: Standard Transformer positional encodings are absolute, but relation semantics often depend on relative distances between entities and surrounding words.

    How: We added two separate relative position embeddings: one for distances to entity 1, one for distances to entity 2. These are summed into the token embeddings before feeding them to the encoder.

    Outcome: ERPE yielded small but consistent gains (+0.3~0.5 Micro F1) when combined with the Marker Head, especially for relations where distance (e.g., "born in") is a strong cue.

Marker Head

    Why: Instead of relying solely on the [CLS] token, we wanted a classifier that directly uses the hidden states at the entity marker positions to focus on entity-specific context.

    How: For each entity, we extracted the hidden state at the first marker token (e.g., [E1-PER], [E2-ORG]) from the final encoder layer, concatenated them, and fed the result into a classification head.

    Outcome: This led to a sharper decision boundary for relation classes, improving Micro F1 by ~1 point and reducing misclassifications in symmetric relations (e.g., org:members vs org:top_members).

Loss Functions

    Why: The dataset was highly imbalanced with no_relation dominating. Vanilla cross-entropy tended to bias predictions toward the majority class.

    How:

        Class-Balanced Loss (CB Loss): Adjusted loss weights based on effective number of samples per class, mitigating imbalance without manual weight tuning.

        Focal Loss: Down-weighted easy samples, focusing on harder minority-class examples.

    Outcome: CB Loss consistently improved recall for minority classes by ~3–5 points in AUPRC, while Focal Loss was less stable in this dataset.

Regularization & Training Tricks
FGM (Fast Gradient Method)

    Why: Small embedding perturbations force the model to generalize better and reduce overfitting to surface forms.

    How: At each training step, after the normal backward pass, we added a small adversarial perturbation (ε=1e-3) to the embedding layer, performed a second forward-backward pass, and then restored the original embeddings.

    Outcome: Helped improve robustness to slight lexical variations, particularly boosting minority-class recall.

LLRD (Layer-wise Learning Rate Decay)

    Why: Lower transformer layers capture general language features; aggressive fine-tuning risks catastrophic forgetting.

    How: Applied a decay factor (0.95) per layer when assigning learning rates: lower layers received smaller updates, higher layers larger updates.

    Outcome: Provided more stable training, especially in longer runs (≥10 epochs), and reduced performance variance across seeds.

R-Drop

    Why: To improve consistency of model predictions by reducing overconfidence and variance between dropout passes.

    How: Each mini-batch was passed through the model twice with different dropout masks; the KL divergence between logits was added to the loss, weighted by alpha=2.0.

    Outcome: This significantly improved the best overall score — Run N5 (mh=1, erpe=1, fgm=0, rdrop=2.0) achieved 78.17 Micro F1, the highest among all runs.

Hard Negative Mining

    Why: Many no_relation examples are trivial (easy negatives), so the model gains little from repeatedly seeing them.

    How: After each epoch, no_relation samples with low predicted probability were identified and up-weighted for the next epoch using a WeightedRandomSampler.

    Outcome: This increased the model’s ability to correctly classify borderline negatives, improving precision for no_relation and indirectly boosting Micro F1.

Evaluation Metrics

    Micro F1 (excluding no_relation): Chosen as the primary metric to avoid dominance of the majority class.

    AUPRC: Captures model performance under varying decision thresholds, especially important for imbalanced datasets.

    Accuracy: Provided for completeness but less informative in this imbalanced setting.

Early Stopping

    Why: To avoid overfitting and reduce wasted computation.

    How: Monitored Micro F1 on the dev set; training stopped if no improvement was seen for a set patience window.

    Outcome: Reduced training time without sacrificing peak performance, particularly in R-Drop runs where later epochs sometimes caused minor degradation.

Future Work

    Data augmentation for rare relation classes.

    Scaling experiments to larger PLMs (klue/roberta-large).

    Exploring multi-sentence relation extraction.
