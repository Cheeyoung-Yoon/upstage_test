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



Describe your approach. You can include EDA (exploratory data analysis), training/evaluation schemes, or summarizations of any literature relevant to this problem. It is desirable that you include rationale behind experimental design and decisions. You can also include future work, which are tasks you planned but could not complete. You are free to use open source software as long as you give attribution.

