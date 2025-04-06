# Transaction Language Model Fine-Tuning

This repository is a workspace for fine-tuning language models to classify personal credit card transactions into predefined budget categories ("from accounts"). The goal is to automate personal finance categorization based on a transaction's details.

## Dataset and Task

- __Source & Privacy:__ The data originates from personal credit card statements. Due to its sensitive nature, the raw dataset is not published.
- __Size:__ The dataset is relatively small, providing a challenging scenario for fine-tuning:
    - Training set: 674 examples
    - Test set: 193 examples
    - Total: 867 examples
        - 93,603 tokens (average example len = 97.8 tokens)
- __Task:__ The objective is multi-class classification – predicting the correct personal budget category (`from_account`) for each transaction.
- __Input Features:__
    - `transaction_date`
    - `description` (vendor name)
    - `amount`
    - `category` (generic category assigned by credit card provider)
    - `category_source` (Discover | Chase)
    - `card` (Discover It Chrome | Chase Sapphire Preferred)
    - `day_of_week`
- __Target Variable (`from_account`):__
    - `Assets:Discover:Furniture`
    - `Assets:Discover:FutureWants`
    - `Assets:Discover:Main:Needs:Gas`
    - `Assets:Discover:Main:Needs:Groceries`
    - `Assets:Discover:Main:Needs:Monthly`
    - `Assets:Discover:Main:Needs:Other`
    - `Assets:Discover:Main:Wants:Monthly`
    - `Assets:Discover:Main:Wants:Other`
- __Example Data Row:__
    ```
    transaction_date : 2023-03-22
    description      : O DONELL ACE HARDWARE DES MOINES IA
    amount           : 23.19
    category         : Home Improvement
    category_source  : Discover
    card             : Discover It Chrome
    day_of_week      : Wednesday
    from_account     : Assets:Discover:Main:Needs:Other  # <- Target Label
    ```

## Results

| Model / Approach                    | Test Accuracy (%) | Date       |
| :---------------------------------- | :---------------- | :--------- |
| GPT-1 (Causal LM Classification)    | 78.7              | March 2025 |
| GPT-1 (Sequence Classification)     | __92.7__          | March 2025 |

### Notes on approaches

#### Definitions

- __Causal LM Classification:__
    - Classification framed as a generative task.
    - Autoregressively generates the category label text token-by-token until `<eos>` is produced.
- __Sequence Classification:__
    - Task-specific classification head is attached to the base model.
    - Directly outputs a prediction across the fixed set of 8 labels.

#### Reflections

The superior performance of Sequence Classification compared to the Causal LM approach likely stems from the hierarchical structure of the category labels. Here's why:

- Causal LM:
    - Causal models generate the label token-by-token (`Assets` -> `:` -> `Discover` -> `:` -> etc.).
    - At each step, the model predicts the next most likely token. An early choice that seems locally probable (like predicting `Main` after `Assets:Discover:`) might lead the generation down a path that prevents it from forming the globally correct, complete label (e.g., if `Assets:Discover:Furniture` was the true label but `Furniture` seemed less likely than `Main` at that specific step).
    - Standard generation methods like greedy search commit to these sequential choices with no backtracking. This problem could be addressed using more advacned generation methods like beam search, but at the cost of many more forward passes.

- Sequence Classification:
    - The classification head evaluates all 8 potential complete labels simultaneously, based on the overall meaning of the input transaction.
    - It directly outputs a score for `Assets:Discover:Main:Needs:Other`, `Assets:Discover:Furniture`, etc., all at once.
    - This avoids the sequential commitment problem, making it easier to identify the best overall category match, even with complex hierarchical labels.
    - In essence, the classification head approach bypasses the tricky step-by-step generation of structured labels, leading to better accuracy for this specific task.

## Repository Structure

- `src/models/`: Contains Python scripts or notebooks for fine-tuning and evaluating each model/approach
- `src/dataset/`: Analysis of non-sensitive aspects of the dataset, such as class distribution, token count, and more.
- `src/dataset/notes/`: Includes code snippets and reflections related to specific ML techniques explored during the project (e.g., LoRA, bfloat16 precision, MPS hardware acceleration).

## Related Work

An alternative approach, __Retrieval-Augmented Classification (RAC)__, was previously explored for this same transaction classification task using an earlier, but similar, version of the dataset. To learn more, check out my blog post: [Retrieval Augmented Classification — LLMs as Classifiers](https://medium.com/the-quantastic-journal/retrieval-augmented-classification-llms-as-classifiers-c28d40391738)