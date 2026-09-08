# Automated Classification of Consumer Financial Complaints

Can a free-text consumer financial complaint be sorted into the right product category automatically, and what does the accuracy actually cost? This project benchmarks TF-IDF + Logistic Regression against a fine-tuned DistilBERT on identical data from the CFPB Consumer Complaint Database, the closest public stand-in for the complaint traffic a bank receives, and proposes a triage structure built on what the comparison shows.

---

## The Result

Both models were trained on the same 58,036 complaints and scored on the same held-out test
set of 14,509. **DistilBERT costs 153× the training time, and a GPU, for 1.7 points of macro F1.**

| Metric | TF-IDF + Logistic Regression | DistilBERT |
|---|---|---|
| **Macro F1** | 0.7973 | **0.8140** |
| Accuracy | 0.8132 | **0.8318** |
| Training time | 7.3 s (CPU) | 18 min 36 s (T4 GPU) |


| Category | LR F1 | DistilBERT F1 |
|---|---|---|
| Student loan | 0.936 | 0.935 |
| Mortgage | 0.919 | 0.928 |
| Vehicle loan or lease | 0.866 | 0.889 |
| Money transfer | 0.844 | 0.850 |
| Debt collection | 0.795 | 0.818 |
| Credit card | 0.793 | 0.799 |
| Checking or savings | 0.779 | 0.789 |
| Payday / personal loan | 0.760 | 0.784 |
| Prepaid card | 0.754 | 0.773 |
| Debt or credit management | 0.528 | 0.576 |

**DistilBERT outperforms LR when lexical usage overlaps.**  The gain concentrates where different categories share similar words (Debt or credit management +0.048, Payday loan
+0.024, Debt collection +0.023). If the categories are lexically distinct, the
baseline is the better buy.

**Neither model separates Debt collection from Debt or credit management.**  
Debt or credit management is the only category below 0.6 F1 in both models. 

**The categories that share vocabulary are the ones that get confused.** 
The model tends to mix up complaint categories whenever they use overlapping vocabulary, because a standard bag-of-words approach only works as long as words remain distinctive. Looking at the data, a category's lexical similarity strongly predicts how often it gets swapped with another (Spearman ρ = 0.67, p = 4e-07). 

![Lexical similarity between category pairs plotted against how often the pair is swapped, Spearman rho 0.67 across all 45 pairs](lexical_similarity_vs_confusion.png)

*Each dot is one pair of categories, 45 in all. Further right means the two categories are
worded more alike; further up means the model swaps them more often.*

---

## Proposed deployment structure

The goal here is to use this benchmark to propose a structured process a
complaints receiving body could adopt.

**Three layers.**

1. **Screen everything with TF-IDF + Logistic Regression.** It trains in 7.3 seconds on a
   CPU and is already accurate on the categories that distinctive vocabulary alone can
   settle.
2. **Send only low-confidence cases to DistilBERT.** When LR's top-class probability falls
   below a threshold, the case goes to the Transformer. That threshold is the dial that
   trades GPU cost against accuracy.
3. **Give every receiving agent one-click reassignment.** A convenient correction path
   makes mistakes tolerable, and each correction is a labelled example for the next
   retraining round.


**Two changes worth considering for the two ambiguous categories.**

1. **Describe each category at the point of selection.** A one-line definition beside each
   option on the intake form resolves the ambiguity where the label is created.
2. **Combine Debt collection and Debt or credit management into one.** Where both already
   route to the same handling team, a single merged category removes the boundary entirely.

---

## Data and sampling

The source is the CFPB Consumer Complaint Database. Only two fields are used: the narrative
text and the product label. After filtering (see the pipeline below), the labelled pool holds
**407,321 complaints** across ten categories, distributed as they occur naturally:

| Product category | Share of pool |
|---|---|
| Debt collection | 33.1% |
| Money transfer, virtual currency, or money service | 17.9% |
| Checking or savings account | 17.0% |
| Credit card | 14.5% |
| Mortgage | 5.0% |
| Vehicle loan or lease | 4.1% |
| Student loan | 3.5% |
| Payday loan, title loan, personal loan, or advance loan | 2.8% |
| Prepaid card | 1.2% |
| Debt or credit management | 0.87% |

Credit reporting was excluded: it alone exceeds all others combined and would have dominated
the task. Sampling then caps each class at **8,000 rows**, eight classes hit the cap,
Prepaid card and Debt or credit management are taken in full, yielding 72,545 rows, split
80/20 with stratification into **58,036 training** and **14,509 test** complaints.

**The cap is applied before the split**, so both sets inherit the flattened distribution.
That ordering is the single most consequential design choice here, and it is why the 38:1
imbalance of the real pool is absent from the evaluation.

---

## Limitations

**The labels are self-reported.**  
Each label is chosen by the consumer filing the complaint, not by a trained annotator,
which sets a ceiling on model accuracy. A bank running the same pipeline over its own
verified, professionally labelled complaints should expect better numbers than the ones
reported here.

**A class average can hide sub-groups.**  
Current similarity metric treats each category as a single average complaint, which masks what is really happening in the data. "Money transfer," for example, lumps two very different issues together: complaints about apps like Zelle or Cash App, which have a highly distinct vocabulary, and bank transfer fraud, which reads exactly like a routine checking or savings problem. Because a broad average doesn't accurately represent either of these specific groups, the metric ends up underestimating the real vocabulary overlap that actually causes the model to make mistakes.

**The evaluation set is more balanced than reality.**  
The per-class cap leaves the test set close to uniform while the underlying pool is
imbalanced, so the per-class numbers for rare classes are optimistic. The lesson: an
evaluation set should mirror the distribution the model will actually meet in production.

---

## Pipeline

Run the three notebooks **in order** — each consumes what the previous one produces.

**1. `data_cleaning.ipynb`** — filters the CFPB database from ~17M rows down to
**407,321**, using DuckDB to stream from disk; a file this size exhausts memory in pandas.
Three filters: the ten product categories, a `Date received` range of 2025-01-01 to
2026-12-31, and a non-empty narrative. Input `complaints.csv`; output `cfpb_filtered.parquet`.

**2. `ml_model_lr.ipynb`** — sampling, the 80/20 stratified split, redaction-mask cleaning
(the CFPB replaces personal information with `XXXX` runs and `XX/XX/XXXX` date masks; these
are stripped before lowercasing), TF-IDF (20,000 features, unigrams + bigrams, fitted on the
training set only), and Logistic Regression with `class_weight='balanced'`. Outputs
`train.parquet`, `test.parquet`, the fitted model and vectoriser, a confusion matrix, and
the class-similarity analysis behind the scatter above. **This notebook writes the shared
split** so that both models are scored on the same test set.

**3. `dl_model_distilbert.ipynb`** (Google Colab, GPU required) — fine-tunes
`distilbert-base-uncased` on the same training data and evaluates on the same test set.


---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Not included in the repository (too large for version control): the raw `complaints.csv`
(~9 GB), the filtered parquet, and the train/test split parquets. All are regenerated by
running the notebooks in order.

---

## Data source

Consumer Financial Protection Bureau (2026). *Consumer Complaint Database.*
<https://www.consumerfinance.gov/data-research/consumer-complaints/>

Public U.S. federal government data. Complaint narratives are published with the consumer's
consent and with personal information redacted by the CFPB.
