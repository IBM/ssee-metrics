# Metrics for LLM Evaluation: Structured Semantic Entity Evaluation (SSEE)

Traditional LLM evaluation metrics such as ROUGE, perplexity, and BLEU do not account for the structure of sets (or unordered lists) and primarily focus on surface text representations. Metrics like BLEU and ROUGE use tokens rather than entities for comparisons and do not consider synonyms. To address these limitations, we introduce a collection of interpretable performance evaluation metrics designed to capture both the structure and semantic content of sets.

Our proposed metrics, collectively referred to as Structured Semantic Entity Evaluation (SSEE), are tailored to evaluate any structured content that can be represented as collections of sets (or unordered lists).

#### Key Features of SSEE
- __Entity-Focused__: The evaluation is centered on entities rather than tokens.
- __Semantic__: We evaluate the semantic representation of entities rather than their surface representations.
- __Order-Independent__: SSEE metrics are designed to handle the unordered nature of sets (or unordered lists).


#### Metrics
This repo implements the following metrics:
- __SSEE Recall__: Measures the fraction of entities in the gold set that are correctly identified in the candidate set.
- __SSEE Precision__: The analogue of SSEE Recall for precision.
- __SSEE F1__: The harmonic mean of SSEE Precision and SSEE Recall.

SSEE metrics consider the similarity of entities in the candidate and gold sets within a shared embedding space. Sorting is used to identify the closest matches between the gold and candidate entities, regardless of order.
A more detailed description of these metrics is provided in [1].

## When Should I use SSEE metrics?
If you want to compare two sets or lists of strings for semantic similarity, SSEE metrics can be of use in the evaluation of your application.  SSEE metrics are order agnostic and use semantic rather than surface similarity to compare lists or sets of elements.


## How to Install
1. Clone the Repository.
2. Navigate into the cloned repository.
```sh
cd ssee_metrics
```
3. Run ```pip install .``` to install dependencies and ssee_metrics package.

## Example Usage
This assums that the ssee_metrics package has been installed as explained in the previous section.
### Calculating Recall
Input
```python
from ssee_metrics.recall import structured_semantic_entity_evaluation_recall as ssee_recall
from ssee_metrics.semantic_utils import SentenceTransformerSemanticUtils

E_gold = ["Apple", "Orange", "Lemon"]
E_cand = ["Plum", "Grape", "Melon"]
T = 1.0
stsu = SentenceTransformerSemanticUtils()
recall = ssee_recall(E_cand, E_gold, T, stsu)
print(recall)
```
Output

```python
0.0
```


### Calculating Precision
Input
```python
from ssee_metrics.precision import structured_semantic_entity_evaluation_precision as ssee_precision
from ssee_metrics.semantic_utils import SentenceTransformerSemanticUtils

E_gold = ["apple", "banana", "cherry"]
E_cand = ["apple", "pineapple", "banana"]
T = 0.9
stsu = SentenceTransformerSemanticUtils()

precision = ssee_precision(E_cand, E_gold, T, stsu)
print(precision)
```

Output

```python
0.6666666666666666
```

## Testing
To run the unit tests for this project, navigate to the project directory and execute the following command:

```sh
python -m unittest discover
```


## References
[1] Lynch, K., Lorenzi, F., Sheehan, J. D., Kabakci-Zorlu, D., & Eck, B. (2025). Structured Document Generation for Industrial Equipment. Proceedings of the AAAI Conference on Artificial Intelligence, 39(28), 28850-28856. https://doi.org/10.1609/aaai.v39i28.35150