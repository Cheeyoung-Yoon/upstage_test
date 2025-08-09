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
So for tokenizer, 

| Tokenizer                                                | Reason it matches well                                                                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **`klue/roberta-base` tokenizer**  (if roberta-base is the best, then try klue/roberta-large)                      | Same as above but lighter. Still matches your corpus style well.     |
| **`monologg/koelectra-base-v3-discriminator` tokenizer** | Korean-specific SentencePiece; good for keeping entities whole; more efficient.                                                |
| **`snunlp/KR-ELECTRA-discriminator` tokenizer**          | Massive Korean corpus coverage; good for formal names like in your dataset.                                                    |
| **`kykim/bert-kor-base` tokenizer**                      | WordPiece Korean vocab; solid baseline for BERT-style; may split some entities more.                                           |

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

