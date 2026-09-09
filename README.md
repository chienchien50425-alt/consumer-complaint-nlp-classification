# Automated Classification of Consumer Financial Complaints

## Goal

Can a free-text consumer complaint be sorted into the right product category automatically,
and what does the accuracy cost?

This benchmarks an interpretable baseline — TF-IDF + Logistic Regression — against a
fine-tuned DistilBERT on the CFPB Consumer Complaint Database, across ten product
categories. Both models train on the same rows, select on the same validation set, and are
scored once on the same test set.

The secondary question is the one that turned out to matter more: **how much of a
classification score is an artefact of how the data was split?** The corpus is
template-driven and its class mix is not stationary, so the evaluation design is reported
here in as much detail as the model.

---

## Results

Logistic Regression, 10.0 s on CPU, scored on 50,199 held-out complaints at the pool's real
28.9:1 class imbalance.

| | text diversity | intake volume |
|---|---|---|
| **Macro F1** | **0.7147** | 0.7052 |
| Accuracy | 0.7963 | 0.7899 |
| Weighted F1 | 0.8136 | |
| Balanced accuracy | 0.7854 | |

Two views of the same test set. *Text diversity* counts each distinct complaint once — can
the model read complaints? *Intake volume* weights each one by how often it was actually
filed (`sample_weight=n_copies`) — can it handle the queue? Template letters arrive
thousands of times in production, and no duplicate enters training under either view.

*DistilBERT is being measured on this split; its column is filled in when that run
completes.*

**Floors, so the headline can be read:**

| Baseline | accuracy | macro F1 |
|---|---|---|
| Always predict the largest class | 0.294 | **0.045** |
| Random draw from the class prior | 0.182 | 0.102 |
| **Logistic Regression** | **0.796** | **0.715** |

Macro F1 is the headline because all ten routing queues matter equally regardless of the
traffic each carries — a claim about the decision, not about the metric. The accuracy floor
of 0.294 looks respectable and is worthless; the comparison that means something is 0.045
against 0.715.

### Per-class

| Class | F1 | support |
|---|---|---|
| Mortgage | 0.920 | 3,071 |
| Student loan | 0.915 | 2,088 |
| Debt collection | 0.862 | 14,777 |
| Credit card | 0.812 | 8,423 |
| Money transfer | 0.806 | 6,714 |
| Checking or savings | 0.793 | 9,696 |
| Vehicle loan or lease | 0.790 | 2,486 |
| Payday / personal loan | 0.597 | 1,681 |
| Prepaid card | 0.450 | 751 |
| Debt or credit management | 0.200 | 512 |

Support is in the table because it spans 512 to 14,777 and the F1 column cannot be read
without it. At n = 512 the 95% confidence half-width on a per-class F1 is roughly **±0.035**,
so small per-class differences are not findings.

The weak classes fail on **precision, not recall**:

| Class | precision | recall | F1 |
|---|---|---|---|
| Debt or credit management | **0.123** | 0.535 | 0.200 |
| Prepaid card | **0.317** | 0.774 | 0.450 |
| Payday / personal loan | 0.493 | 0.757 | 0.597 |

`class_weight='balanced'` fits the model to a uniform class prior. Released onto a 28.9:1
test set it still finds the rare classes — recall holds — but drowns them in false positives.
Debt collection alone sends 1,250 complaints to Debt or credit management, a class with only
512 true rows in the entire test set.

### Correcting for the prior shift is a trade, not a free win

Because the fitted prior is uniform, label-shift correction reduces to adding the log
evaluation prior to the logits (the `− log π_train` term is constant and drops out of the
softmax):

    z'_k = z_k + log π_eval(k)

Selected on validation, applied once to test:

| Operating point | macro F1 | accuracy | balanced accuracy |
|---|---|---|---|
| Uniform-prior | 0.7147 | 0.7963 | **0.7854** |
| Prior-corrected | **0.7414** | **0.8297** | 0.7033 |

Correction buys +0.027 macro F1 and +0.033 accuracy, and costs −0.082 balanced accuracy. The
corrected model stops over-predicting rare classes, which lifts precision and therefore F1 —
but it also stops finding as many of them. Which point is right depends on whether a missed
rare complaint or a mis-routed common one is the more expensive error, which the data cannot
settle.

### The training cap costs 0.014 macro F1

Training is capped at 8,000 rows per class as a **compute budget**, not as imbalance
handling — `class_weight='balanced'` does that work independently. The cap's only effect is
deleting 164,523 labelled rows:

| Training set | rows | fit | macro F1 | accuracy |
|---|---|---|---|---|
| Capped at 8,000/class | 69,737 | 10.0 s | 0.7147 | 0.7963 |
| Full training pool | 234,260 | 38.0 s | **0.7284** | **0.8135** |

For Logistic Regression the cap buys 28 seconds and is not worth it. It exists because
DistilBERT on the full pool is roughly 80 minutes of GPU rather than 25.

### Held-out months are harder, and one class carries all of it

A random split grades the model on the average of 2025-01 to 2026-07, and **no month looks
like that average**. Money transfer is 42.1% of 2025Q1 and 4.5% of 2026Q3, largely because
of one filing surge over 2025-01-15/20 that peaked at 16,424 complaints in a single day and
accounts for 57% of the entire class.

| Split | test rows | macro F1 | accuracy |
|---|---|---|---|
| Random (pooled 19 months) | 50,199 | 0.7147 | 0.7963 |
| Temporal (unseen months) | 22,020 | **0.6998** | 0.7913 |

The aggregate gap is small, −0.015. It is not evenly spread, and it lands on the class the
surge inflated:

| Class | random | temporal | Δ |
|---|---|---|---|
| **Money transfer** | 0.806 | **0.655** | **−0.151** |
| Prepaid card | 0.450 | 0.391 | −0.059 |
| Credit card | 0.812 | 0.829 | +0.017 |
| Vehicle loan | 0.790 | 0.841 | +0.051 |

A single aggregate would have reported "temporal costs 1.5 points" and hidden that one class
lost 15.

### Why those classes get confused

Each class is represented as the mean TF-IDF vector of its training complaints; the cosine
between two centroids measures shared vocabulary. Plotted against how often the pair is
actually swapped, across all 45 pairs:

**Spearman ρ = 0.806, p = 2.3e-11 (n = 45 pairs)**

![Lexical similarity between category pairs plotted against how often the pair is swapped](lexical_similarity_vs_confusion.png)

Vocabulary overlap predicts confusion strongly. Debt collection / Debt or credit management
are the most lexically similar pair (0.766) and among the most swapped (8.8%).

**The exception is informative.** Checking/savings and Money transfer are the *most*-swapped
pair (10.2%) while sitting only 0.533 apart — off the trend. A centroid is an average, and an
average only describes a class that is one thing. Of 6,714 Money transfer complaints in the
test set, 786 go to Checking/savings, and they sit closer to the other class's centroid than
to their own:

| Group | → Money transfer centroid | → Checking/savings centroid |
|---|---|---|
| Classified correctly | 0.306 | 0.131 |
| Sent to Checking/savings | **0.114** | **0.206** |

Clustering the class on its text alone — not on whether the model got it right, which would
be circular — splits it into a general-language group and two template campaigns:

| Cluster | top terms | train / test | misroutes |
|---|---|---|---|
| 0 | account, money, app, cash, paypal, funds, bank | 5,618 / 4,672 | **786 of 786** |
| 1 | cash app, unfair, resolution, act, fraud platform | 1,385 / 1,145 | 0 |
| 2 | zelle, issues platform, handling disputes, cfpb lawsuit | 997 / 897 | 0 |

**Every misroute comes from the general-language cluster; the two campaign clusters make
none.** Complaints written in plain financial language — money, account, bank, funds — read
exactly like a checking complaint, because at the word level they are one.

A caveat on method: the silhouette score does not choose k here. It never peaks, rising from
0.112 at k=2 to 0.167 at k=8, and every value in that band is weak structure. k=3 is not an
elbow — it is the smallest k at which the two campaigns separate. What is stable is the
*error concentration*: the misroutes sit in one cluster at every k tested.

---

## Method

### Splitting

Three properties the split has to have, and none of them are automatic on this corpus.

**1. No text may straddle two splits — including after cleaning.** The pool is
template-driven: 104 distinct texts are filed 50 or more times each, and the largest single
template appears 7,111 times. A row-wise random split scatters copies of one letter across
train and test, so the model is graded on text it memorised.

Deduplicating on the raw narrative is not sufficient. The model never sees the raw string —
it sees `clean_text(narrative)`, which strips digits, punctuation and case, so two narratives
differing only in a redacted amount collapse to the same input. That adds **8,374 further
collisions covering 11,398 rows**, and it is asymmetric: Logistic Regression's cleaner is
more aggressive than DistilBERT's, so LR would retain roughly 3.3× more residual memorisation
than the model it is compared against.

So deduplication, conflict detection and splitting all key on the **normalised** text. Two
documents the model cannot tell apart are one document to the splitter.

**2. Evaluation is never rebalanced.** Training data may be; test data may not. The per-class
cap is applied to the training pool only, after the evaluation sets are held out at the
pool's own distribution.

**3. Model selection never touches the test set.** A held-out validation set chooses the
DistilBERT checkpoint and the Logistic Regression operating point. The test set is read once.

```
407,321 rows filtered from the raw database
  −     22  normalise to an empty string
  − 24,007  carry conflicting labels under the same normalised text (210 distinct texts)
  − 48,634  duplicate copies, removed on the normalised key
= 334,658  distinct complaints, imbalance 28.9:1

  ├─ [primary]  random stratified 70 / 15 / 15
  │             train pool 234,260 → capped 69,737 · val 50,199 · test 50,199
  │
  └─ [secondary] temporal
                train ≤ 2026-02 (276,023 → capped 71,132)
                val    2026-03/04 (36,615)
                test   2026-05/07 (22,020)
```

Class shares are preserved in both evaluation sets and never capped:

| Product category | pool | random test | temporal test |
|---|---|---|---|
| Debt collection | 29.44% | 29.44% | 26.04% |
| Checking or savings account | 19.31% | 19.32% | 22.27% |
| Credit card | 16.78% | 16.78% | 20.93% |
| Money transfer, virtual currency, money service | 13.37% | 13.37% | **8.24%** |
| Mortgage | 6.12% | 6.12% | 7.84% |
| Vehicle loan or lease | 4.95% | 4.95% | 5.75% |
| Student loan | 4.16% | 4.16% | 3.05% |
| Payday / title / personal loan | 3.35% | 3.35% | 3.74% |
| Prepaid card | 1.50% | 1.50% | 1.34% |
| Debt or credit management | 1.02% | 1.02% | 0.80% |

### Models

**TF-IDF + Logistic Regression.** Redaction masks stripped before lowercasing (the CFPB's
`X` runs are uppercase), then case and non-alphabetic characters dropped. TF-IDF at 20,000
features, unigrams + bigrams, fit on train only. `LogisticRegression(class_weight='balanced')`
at scikit-learn defaults.

**DistilBERT.** `distilbert-base-uncased`, 3 epochs, batch 32, learning rate 2e-5, max 256
wordpieces. Cleaning is lighter — digits, case and punctuation are kept, because a Transformer
can use them. Loss is class-weighted with the same `N / (K · n_k)` weights scikit-learn's
`'balanced'` uses, so imbalance is handled identically on both sides.

Neither model is tuned. Both use one fixed configuration, so neither gets a search advantage.

### Comparing the two

Both models score the same test rows, so the comparison is **paired**: a paired bootstrap
(2,000 resamples) gives a confidence interval on the macro-F1 difference, and McNemar's test
on the per-row hits gives a p-value for the accuracy difference. Marginal per-class intervals
are too wide at this support to resolve small differences on their own.

---

## Limitations

**Only exact duplicates are removed.** Deduplication uses an exact normalised key. 20,824
rows share their first 200 characters with another row, so near-duplicate templates survive
into the splits. A MinHash pass at Jaccard ≥ 0.8 would be the correct treatment and is not
implemented. The claim this repo supports is *no exact-key leakage*, not *no leakage*.

**Deduplication changes what "the real distribution" means.** Duplicates are concentrated in
two classes — 33.4% of Money transfer rows and 24.1% of Debt collection are removed, against
under 6% everywhere else. The resulting pool describes the distribution of *distinct
complaints*, not of intake volume, which is why every headline is reported both ways.

**The labels are self-reported.** Each label is chosen by the consumer filing the complaint,
not by a trained annotator. 210 distinct texts carry more than one product label, but that is
not a label-noise rate: 96.8% of the rows involved come from 17 mass-filed templates where a
handful of filers picked a different drop-down, and after deduplication those 210 texts are
0.06% of the pool. What is suggestive is *which* pairs disagree — Credit card ↔ Debt
collection (44 texts), Checking/savings ↔ Money transfer (24), Debt collection ↔ Vehicle loan
(18) — the same pairs the model confuses. Qualitative corroboration, not a measured ceiling.

**The temporal test set is small and its tail is incomplete.** 22,020 rows, of which 176 are
the rarest class, so its per-class F1 is noisy. 2026-06 and 2026-07 hold 6,139 and 1,756
complaints against a monthly norm near 18,000 — the database is still filling in. The date
filter's upper bound (2026-12-31) is in the future, so the pool is not a fixed window but
whatever had been published at download time; the last complaint is dated 2026-07-17.

**Ten flat categories is a simplification.** Real complaint taxonomies are hierarchical and
far larger; one documented bank deployment uses a five-level, ~4,000-code frame.

**The two pipelines are not preprocessed identically.** LR lowercases and strips digits and
punctuation; DistilBERT keeps them. DistilBERT truncates at 256 wordpieces while TF-IDF reads
every document in full. Class weighting is matched. The remaining differences are disclosed
rather than removed, and they are part of what "the cost of a Transformer" means here.

**Neither model is tuned**, so neither number is the best that architecture could reach.

**`data_cleaning.ipynb` has no stored outputs.** It cannot be executed without the raw
`complaints.csv` (~9 GB), which is not in the repository. Everything downstream of
`cfpb_filtered.parquet` carries its outputs.

---

## Data pipeline

Run the notebooks in order — each consumes what the previous one produces.

**1. `data_cleaning.ipynb`** — filters the raw CFPB export down to 407,321 rows using DuckDB
to stream from disk (the file exhausts memory in pandas). Three filters: the ten product
categories, `Date received` in 2025-01-01 … 2026-12-31, and a non-empty narrative. Credit
reporting is excluded — it alone exceeds all other categories combined.
Input `complaints.csv`; output `cfpb_filtered.parquet`. The date filter's upper bound is a
future date, so re-running against a fresher export produces a larger pool and different
numbers throughout.

**2. `ml_model_lr.ipynb`** — **owns the split.** Canonical-key deduplication, conflict
removal, multiplicity, the random and temporal splits, TF-IDF, Logistic Regression, the floor
baselines, the prior-correction and cap ablations, the lexical-similarity analysis, and the
paired significance tests. Writes `train/val/test.parquet` and `*_temporal.parquet`.

**3. `dl_model_distilbert.ipynb`** (Colab, GPU) — fine-tunes `distilbert-base-uncased` on the
same training set, selects its checkpoint on the same validation set, and reads the same test
set once. Writes `distilbert_test_predictions.csv`, which Step 9 of notebook 2 consumes for
the paired bootstrap and McNemar's test.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Not in the repository (too large for version control): the raw `complaints.csv`, the filtered
parquet, and the split parquets. All are regenerated by running the notebooks in order.

### Data source

Consumer Financial Protection Bureau. *Consumer Complaint Database.*
<https://www.consumerfinance.gov/data-research/consumer-complaints/>

Public U.S. federal government data. Narratives are published with the consumer's consent and
with personal information redacted by the CFPB.
