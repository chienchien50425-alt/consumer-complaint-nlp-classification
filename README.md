# Auto-sorting Financial Complaints
> How accurately can free-text complaints be automatically classified for routing? This project evaluates the predictive performance and practical trade-offs of different NLP approaches using real-world consumer complaints


## Goal

This project wants to build and evaluate an automated complaint-routing system that assigns customer complaints to the correct product category. The project compares a traditional TF-IDF + Logistic Regression model with DistilBERT to determine whether advanced NLP delivers enough improvement in classification accuracy to justify its complexity and computational cost.

Model performance will be evaluated both overall and by complaint   category to identify strengths, weaknesses, and limitations.


## Results

### DistilBERT wins every metric, but by margins 1.1-2.0 points, and it takes far longer time to train. Both models were scored on the identical 36,615-row test set.


| Metric | TF-IDF + Logistic Regression | DistilBERT |
| --- | --- | --- |
| Accuracy | 0.7898 | 0.8034 |
| Macro F1 | 0.7030 | 0.7225 |
| Weighted F1 | 0.8061 | 0.8173 |
| Training time | **11.6 s**, CPU | 1,434 s, GPU |

> **Macro F1** averages the ten per-category F1 scores equally.  
> **Weighted F1** averages the ten per-category F1 by category size.


### Both models are strongest and weakest on exactly the same categories

| Category | Test rows | LR | DistilBERT | Gain (F1 pts) |
| --- | --- | --- | --- | --- |
| Mortgage | 2,641 | 0.929 | 0.935 | +0.6 |
| Student loan | 1,459 | 0.918 | 0.923 | +0.5 |
| Debt collection | 10,706 | 0.871 | 0.876 | +0.5 |
| Vehicle loan or lease | 2,050 | 0.829 | 0.857 | +2.8 |
| Credit card | 6,861 | 0.815 | 0.827 | +1.2 |
| Checking or savings | 7,201 | 0.775 | 0.782 | +0.7 |
| Money transfer / crypto | 3,455 | 0.697 | 0.724 | +2.7 |
| Payday / personal loan | 1,430 | 0.617 | 0.629 | +1.2 |
| Prepaid card | 431 | 0.345 | 0.419 | +7.4 |
| Debt or credit management | 381 | 0.236 | 0.255 | +1.9 |


- Mortgage and Student loan are near-solved. Debt or credit management and Prepaid card are badly broken for both.

- For category owning distinctive vocabulary, TF-IDF has already captured everything.

### The verdict

DistilBERT is ahead by 2.0 points of macro F1 and costs roughly 124 times the training time and additional GPU resources. Either way, two points of macro F1 concentrated in categories that
remain unusable afterward is not what a routing team buys a GPU for. On this task, with
these labels, the two are a practical tie.

### Why those categories share a wall

The confusion matrix says which categories get mixed up. It does not say why. Representing
each category as the mean TF-IDF vector of its training complaints, the cosine between two
centroids measures how much vocabulary the pair shares. Plotting that against how often the
pair actually gets swapped turns "these categories sound alike" into a testable claim.

Across all 45 category pairs, shared vocabulary predicts swap rate: **Spearman rho = 0.685,
p = 2.1e-07**. Similarity is measured on training data and the error rate on held-out test
data, so the two axes come from disjoint sources. The worst pair, Checking or savings
against Money transfer, swaps 1,275 complaints, 12.0% of the two categories combined.

![Lexical similarity against confusion rate](lexical_similarity_vs_confusion.png)

This names the wall. Shared vocabulary is what holds those categories down, and it is also
exactly what contextual reading partially repairs — which is why the gains in the table
above fall where they do and nowhere else.

The pattern generalises past finance. Any company whose categories share terminology, whose
volumes are lopsided, and whose labels are chosen by the customer will see the same shape:
a transformer that pays for itself only on the overlapping queues, and pays for nothing on
the clean ones.

## Key configuration

Everything below the model family was deliberately matched. Where the two rows differ, the
difference is a property of the method, not a tuning choice.

| | TF-IDF + Logistic Regression | DistilBERT |
| --- | --- | --- |
| Text cleaning | Redaction masks, case, punctuation and digits all stripped | Redaction masks only — case and punctuation are signal a transformer can use |
| Representation | 20,000 unigram + bigram features, English stopwords removed, fitted on train only | `distilbert-base-uncased`, 256 wordpieces |
| Imbalance handling | `class_weight='balanced'` | Class-weighted cross-entropy, same `N / (K · n_k)` formula |
| Training rows | 70,114 | 70,114 |
| Hyperparameter search | None — scikit-learn defaults (`C=1.0`, L2, lbfgs) | None — 3 epochs, lr 2e-5, batch 32, weight decay 0.01 |
| Checkpoint selection | Not applicable | Best macro F1 on the validation set |
| Hardware and time | CPU, 11.6 s | Tesla T4, 1,434 s |

The cleaning asymmetry is intentional and it is the one place the pipelines genuinely
diverge. It also forces a constraint on the split, covered next.

## The data and the pipeline

The source is the CFPB Consumer Complaint Database, a ~9.1 GB CSV of ~17M rows, streamed
with DuckDB rather than loaded into memory. Filtering to complaints received between
2025-01-01 and 2026-12-31, in ten product categories, with a non-empty narrative leaves
**407,321 rows** at a 38.2x imbalance ratio.

The corpus is template-driven: the single most-filed text appears 7,111 times. Splitting
at random puts copies of one letter on both sides of the split and scores the model on text
it memorised. So rows are deduplicated on a canonical key that applies the same
normalisation the model input goes through — which has to be at least as aggressive as the
more aggressive of the two cleaners, or documents the model cannot tell apart still
straddle the split. 210 keys carry conflicting labels (24,007 rows) and are dropped
outright; the rest collapse to **334,658 distinct complaints**.

The split is then time-ordered, training on the past and evaluating on months never seen:

    train  ≤ 2025-12  (capped to 70,114)
    val     2026-01/02  (29,705)
    test    2026-03/04  (36,615)

A further 22,020 rows from 2026-05 onward are excluded rather than scored. The CFPB
publishes a narrative only after the company responds or 60 days pass, so fast responders
appear before slow ones and those months are biased, not merely incomplete.

Three rules hold throughout: no text straddles a split even after cleaning; evaluation sets
are never rebalanced, so the test set runs at its real 28:1 imbalance; and the test set is
read exactly once, with all model selection done on validation. One cost is worth naming:
because deduplication runs before the date cut, templates filed in both windows may survive
only in training, leaving the test set skewed toward genuinely novel text.

## Limitations

**The labels were chosen by consumers, not by specialists.** Every category in this dataset
was picked from a drop-down by the person filing the complaint. "Debt collection" versus
"Debt or credit management" is a distinction a consumer has no particular reason to draw
correctly, and those two are simultaneously the most lexically similar pair in the corpus
(cosine 0.772) and among the worst-scoring for both models. Some fraction of what is
measured here as model error is label noise. On a specialist-annotated corpus both models
should score higher, and the gap between them might well look different too. The ceiling
reported above belongs partly to the data.

**Compute budget, truncation, and no tuning.** Training rows were capped at 8,000 per class
as a GPU budget — imbalance is handled by class weighting, not by the cap. DistilBERT
truncates at 256 wordpieces while TF-IDF reads every document in full, so the two models do
not see the same amount of each long complaint. And neither model received any
hyperparameter search, so neither number is the best its method can do. They are comparable
to each other, not to a tuned benchmark.

## Files and how to run

| File | Role |
| --- | --- |
| `data_cleaning.ipynb` | Filters the raw CFPB CSV with DuckDB, writes `cfpb_filtered.parquet` |
| `ml_model_lr.ipynb` | Owns the split — writes `train/val/test.parquet` — then trains and evaluates Logistic Regression |
| `dl_model_distilbert.ipynb` | Fine-tunes DistilBERT on those same three parquets; needs a GPU (built for Colab with a T4) |
| `requirements.txt` | Dependencies for the two local notebooks |

Run them in that order. The split lives in the second notebook, so the third cannot be run
until it has been.
