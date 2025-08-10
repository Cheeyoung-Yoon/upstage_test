# upstage_test
upstage online deep learning code test 


# 1. Summary
Please summarize the task, and what you have done to tackle the task in five sentences or less.

# 2. Experimental results
List all important experimental results in a table format.

# 3. Instructions
Describe your environment settings, structure of your code, and usage instructions.

# 4. Approach

## Tokenizer:
 Tokenizer is one of the key point on this task. 
 as how tokens are split and represented directly impacts how well the model learns entity relationships. As tokenizer controls how the model 'sees' the entities and their context.
 klue/bert-base uses the original BERT architecture, which may have weaker contextual representation compared to newer Korean PLMs (e.g., RoBERTa, ELECTRA).
   - RoBERTa models remove NSP and train with larger batches, often yielding better contextual understanding and higher accuracy in RE/classification tasks.
   - ELECTRA models are more sample-efficient, potentially delivering better generalization on the same dataset.
So for tokenizer, below is the result of the test for just adding the markers from the raw data.


| Model                                     | lrs   | epochs | train\_bsz | Best Step | Micro F1 | AUPRC   | Accuracy | Val Loss |
| ----------------------------------------- | ----- | ------ | ---------- | --------- | -------- | ------- | -------- | -------- |
| klue\_bert-base                           | 3e-05 | 10     | 16         | 4500      | 59.8642  | 47.9678 | 0.613181 | 1.274938 |
| klue\_bert-base                           | 3e-05 | 5      | 16         | 4000      | 59.7681  | 47.1765 | 0.606098 | 1.287195 |
| klue\_bert-base                           | 5e-05 | 5      | 16         | 7000      | 59.6718  | 47.4565 | 0.609794 | 1.430602 |
| klue\_bert-base                           | 3e-05 | 5      | 32         | 3000      | 59.4809  | 46.9674 | 0.607330 | 1.329728 |
| klue\_bert-base                           | 5e-05 | 5      | 32         | 3500      | 59.4452  | 48.2639 | 0.608870 | 1.387745 |
| klue\_bert-base                           | 5e-05 | 10     | 16         | 5000      | 59.6273  | 47.5093 | 0.604866 | 1.282922 |
| klue\_bert-base                           | 5e-05 | 10     | 32         | 2000      | 59.3649  | 47.0007 | 0.602402 | 1.327961 |
| klue\_roberta-base                        | 5e-05 | 5      | 16         | 7500      | 59.6945  | 47.8759 | 0.604866 | 1.481126 |
| klue\_roberta-base                        | 5e-05 | 5      | 32         | 4500      | 59.8912  | 47.5600 | 0.614721 | 1.428611 |
| klue\_roberta-base                        | 3e-05 | 5      | 16         | 7500      | 60.2211  | 46.0446 | 0.615953 | 1.460129 |
| klue\_roberta-base                        | 3e-05 | 5      | 32         | 3500      | 60.3333  | 47.9823 | 0.615953 | 1.291947 |
| **klue\_roberta-base**                        | 2e-05 | 5      | 16         | 8000      | **60.6296**  | 48.4109 | **0.618417** | 1.374527 |
| klue\_roberta-base                        | 2e-05 | 5      | 32         | 4000      | 58.8595  | 48.3168 | 0.607022 | 1.319022 |
| monologg\_koelectra-base-v3-discriminator | 5e-05 | 5      | 16         | 6000      | 59.3478  | 41.7290 | 0.604558 | 1.384946 |
| monologg\_koelectra-base-v3-discriminator | 5e-05 | 5      | 32         | 3000      | 59.0235  | 39.5930 | 0.607330 | 1.359320 |
| monologg\_koelectra-base-v3-discriminator | 3e-05 | 5      | 16         | 8000      | 59.6228  | 39.8815 | 0.607946 | 1.417223 |
| monologg\_koelectra-base-v3-discriminator | 3e-05 | 5      | 32         | 3000      | 57.6611  | 34.6298 | 0.588235 | 1.338130 |
| bert-base-multilingual-cased              | 5e-05 | 5      | 16         | 5500      | 56.0200  | 41.4280 | 0.579920 | 1.363017 |
| bert-base-multilingual-cased              | 5e-05 | 5      | 32         | 3500      | 57.5140  | 42.7060 | 0.590699 | 1.343769 |
| bert-base-multilingual-cased              | 3e-05 | 5      | 16         | 7500      | 57.3580  | 43.1300 | 0.586387 | 1.438589 |
| bert-base-multilingual-cased              | 3e-05 | 5      | 32         | 2500      | 57.3490  | 43.3980 | 0.581152 | 1.312771 |
| bert-base-multilingual-cased              | 2e-05 | 5      | 16         | 7000      | 57.7780  | 43.6700 | 0.592239 | 1.322330 |
| bert-base-multilingual-cased              | 2e-05 | 5      | 32         | 4500      | 57.9570  | 44.6230 | 0.593163 | 1.326897 |
| kykim\_bert-kor-base                      | 5e-05 | 5      | 16         | 4500      | 59.2880  | 47.0530 | 0.605482 | 1.279820 |
| kykim\_bert-kor-base                      | 5e-05 | 5      | 32         | 3500      | 59.1060  | 44.4160 | 0.606098 | 1.369532 |
| kykim\_bert-kor-base                      | 3e-05 | 5      | 16         | 4500      | 59.1750  | 45.5350 | 0.602710 | 1.268549 |
| kykim\_bert-kor-base                      | 3e-05 | 5      | 32         | 4000      | 58.6210  | 44.7010 | 0.601478 | 1.371919 |
| kykim\_bert-kor-base                      | 2e-05 | 5      | 16         | 5000      | 59.7150  | 47.1080 | 0.595627 | 1.280806 |
| kykim\_bert-kor-base                      | 2e-05 | 5      | 32         | 2500      | 58.2770  | 46.2000 | 0.592239 | 1.279668 |

* Steps are vary aas i added early stopping

So I chose **klue.roberta-base** as the tokenizer.


## Marker
Use maker in the preprocess pipe for following reasons.

1. Reduced Search Space (Attention Guidance)
    Markers explicitly specify the target entity spans, allowing the model to focus on the relevant regions instead of the entire sentence.
    This increases the signal-to-noise ratio (SNR), especially in long sentences or those with many entities, leading to fewer false positives.

2. Resolving Multi-Entity / Same-Type Ambiguity
    In sentences containing multiple entities of the same type (e.g., several PER and ORG entities), it can be unclear which pair the model should consider.
    Markers remove this ambiguity, significantly reducing misclassifications caused by incorrect entity pair selection.

3. Implicit Injection of Entity-Type Constraints
    Type-aware markers (e.g., [E1-PER] ... [/E1], [E2-ORG] ... [/E2]) introduce type priors directly into the model’s input.
    Combined with label masking (assigning -∞ to logits of impossible labels based on entity types), this reduces the effective label space, simplifying the classification task.

4. Stabilization Under Class Imbalance
    With a large proportion of no_relation samples, the model may learn to predict negatives based only on global sentence patterns.
    Markers make it explicit that “this specific pair has no relation”, providing a strong negative learning signal.
    When combined with hard negative mining, this results in sharper decision boundaries between positive and negative cases.

5. Mitigating Tokenization Issues (Especially in Korean)
    In Korean, subword tokenization often splits entities into multiple fragments, making boundary recognition difficult.
    Markers explicitly provide entity boundaries as dedicated tokens, reducing segmentation errors and improving span awareness.


Describe your approach. You can include EDA (exploratory data analysis), training/evaluation schemes, or summarizations of any literature relevant to this problem. It is desirable that you include rationale behind experimental design and decisions. You can also include future work, which are tasks you planned but could not complete. You are free to use open source software as long as you give attribution.

