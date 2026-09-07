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
| **Debt or credit management** | **0.528** | **0.576** |

**DistilBERT outperform LR when lexical usage is overlap.** The gain concentrates where different categories share similar words (Debt or credit management +0.048, Payday loan
+0.024, Debt collection +0.023). If the categories are lexically distinct, the
baseline is the better buy.

**Both models fail on predicting Debt collection and Debt or credit management.**
They are the only pair no model could separate.

---

## Proposed deployment structure

The goal here is to use the pilot test to propose a structure process
mid-to-small-sized bank could adopt.

**Three layers.**

1. **Screen everything with TF-IDF + Logistic Regression.** It trains in 7.3 seconds on a
   CPU and is already accurate on the categories that distinctive vocabulary settles.
2. **Send only low-confidence cases to DistilBERT.** Where the confidence of LR below threshold, the case goes to the Transformer. Threshold is the dial that trades GPU cost
   against accuracy.
3. **Give every receiving agent one-click reassignment.** A convinient correction path makes mistakes tolerable and each correction is a labelled example for the next retraining round.


**Two changes worth considering on ambigious categories.** 

1. **Describe each category at the point of selection.** A one-line definition beside each
   option on the intake form resolves the ambiguity where the label is created.
2. **Combine the two categories into one.** Where both already route to the same handling
   team, a single merged category removes the boundary entirely.
---

## Limitations

**The labels are self-reported.**  
Its labels are chosen by consumers rather than assigned by trained annotators, which sets the upper limit for model accuracy. A bank running the same pipeline over its own verified, professionally labelled complaints should expect better numbers than the ones reported here.


## Lesson learned
**The evaluation set is balanced.** 
The per-class cap leaves the test set
close to uniform while the underlying pool is imbalanced, so per-class precision on rare classes is optimistic. Giving me a lesson on "The test subset should always be align with the real world."

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
the task. Sampling then caps each class at **8,000 rows** — eight classes hit the cap,
Prepaid card and Debt or credit management are taken in full — yielding 72,545 rows, split
80/20 with stratification into **58,036 training** and **14,509 test** complaints.

**The cap is applied before the split**, so both sets inherit the flattened distribution.
That ordering is the single most consequential design choice here, and it is why the 38:1
imbalance of the real pool is absent from the evaluation.

---

## Pipeline

Run the three notebooks **in order** — each consumes what the previous one produces.

**1. `data_cleaning.ipynb`** (local) — filters the CFPB database from ~17M rows down to
**407,321**, using DuckDB to stream from disk; a file this size exhausts memory in pandas.
Three filters: the ten product categories, a `Date received` range of 2025-01-01 to
2026-12-31, and a non-empty narrative. Input `complaints.csv` (see [Data
source](#data-source)); output `cfpb_filtered.parquet`. Two settings this particular file
requires are documented in the notebook: `parallel=false`, because narratives contain
newlines inside quoted fields and DuckDB's parallel CSV reader aborts on them, and reading
`Date received` as VARCHAR so a malformed date deep in the file becomes a countable NULL
rather than a hard cast error.

**2. `ml_model_lr.ipynb`** (local) — sampling, the 80/20 stratified split, redaction-mask
cleaning, TF-IDF (20,000 features, unigrams + bigrams, fitted on the training set only), and
Logistic Regression with `class_weight='balanced'`. Outputs `train.parquet`, `test.parquet`,
the fitted model and vectoriser, and a confusion matrix. **This notebook writes the shared
split** — both models must be scored on the same test set or the comparison is meaningless,
so it is generated once here and reused rather than re-derived in Colab.

**3. `dl_model_distilbert.ipynb`** (Google Colab, GPU required) — fine-tunes
`distilbert-base-uncased` on the same training data and evaluates on the same test set. Set
`Runtime → Change runtime type → T4 GPU` first; on CPU this takes hours instead of ~19
minutes. Upload `train.parquet` and `test.parquet` when prompted; the notebook reads them
from `/content/sample_data/`.

---

## Preprocessing: why the two pipelines differ

The CFPB's redaction masks are the largest source of noise in the corpus. The patterns were
identified by inspecting actual narratives rather than assumed: plain runs (`XXXX`), full
date masks (`XX/XX/XXXX`), scrubber artifacts (`XX/XX/year>`, `XX/XX/scrub>`), partly-real
dates (`XX/XX/2025`), and truncated masks (`XX/XX/`). Masks are stripped *before* lowercasing,
since the `X`s are uppercase.

Beyond mask removal the pipelines diverge deliberately. The **TF-IDF** side cleans heavily —
lowercase, strip non-alphabetic characters, remove English stopwords — because TF-IDF treats
every surface form as a separate feature, so that variation inflates the vocabulary without
adding signal. The **DistilBERT** side removes masks and nothing else: its subword tokenizer
handles casing and punctuation, and those carry meaning it uses. Applying the TF-IDF cleaning
would strip exactly the signal that distinguishes it from a bag-of-words model.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

The Colab notebook installs its own dependencies and needs nothing locally.

**Reproducibility.** `RANDOM_STATE = 42` is fixed for sampling, splitting, and training. Each
model was trained once, so the 1.7-point macro F1 gap carries no confidence interval;
repeated runs across seeds would be needed to confirm it is stable rather than run-to-run
variance.

Not included in the repository (too large for version control): the raw `complaints.csv`
(~9 GB), the filtered parquet, and the train/test split parquets. All are regenerated by
running the notebooks in order.

---

## Data source

Consumer Financial Protection Bureau (2026). *Consumer Complaint Database.*
<https://www.consumerfinance.gov/data-research/consumer-complaints/>

Public U.S. federal government data. Complaint narratives are published with the consumer's
consent and with personal information redacted by the CFPB.
