# TermMap

A clinical term mapping engine that assigns an **ICD-10-CM** code and a **SNOMED CT** concept to terms
extracted from unstructured medical charts.

Given a clinical entity such as `chronickidney disease stag 3` or `HTN` and the chunk of text it came from,
TermMap returns one ICD-10-CM code and one SNOMED CT concept id, each with a confidence score. Both systems are
predicted independently. Neither is derived from the other through a crosswalk.

```
$ python -m termmap map "htn"
ICD10CM      I10          0.955  Essential (primary) hypertension
SNOMEDCT_US  38341003     0.902  High blood pressure

$ python -m termmap map "chronickidney disease stag 3"
ICD10CM      N18.30       0.328  Chronic kidney disease, stage 3 unspecified
SNOMEDCT_US  433144002    0.279  Chronic kidney disease stage 3
```

| Held-out exact match | ICD-10-CM | SNOMED CT |
|---|---|---|
| All terms | **68.9%** | **53.8%** |
| Gold code never seen in training (long tail) | 45.1% | 42.2% |

Documentation: [`docs/TermMap_Documentation.pdf`](docs/TermMap_Documentation.pdf) &nbsp;|&nbsp; Demo video:

---

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Setup](#setup)
- [Running the engine](#running-the-engine)
- [Demo app](#demo-app)
- [GPU steps](#gpu-steps)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Design notes](#design-notes)
- [External resources](#external-resources)

---

## How it works

TermMap is a retrieve-then-rerank cascade. Retrieval is tuned for recall and the rerankers make the decision.

```
          clinical_entity + chunk_text
                       |
           +-----------v------------+
           |  Normalisation         |   case, punctuation, markup, OCR debris
           |  Abbreviation variants |   HTN -> hypertension; MS -> {multiple sclerosis,
           +-----------+------------+          mitral stenosis, mental status}
                       |
   +-----------+-------+-----+-------------+-------------+-------------+
   |           |             |             |             |             |
 char 3-4    word 1-2     training       SapBERT      BioLORD-2023   ICD default
 gram        gram         term           dense        dense          child
 TF-IDF      TF-IDF       aliases                                    expansion
   |           |             |             |             |             |
   +-----------+-------+-----+-------------+-------------+-------------+
                       |
            candidate set (<= 60 codes)
                       |
           +-----------v-----------+
           |  Feature extraction   |  65 features per candidate
           +-----------+-----------+
                       |
           +-----------v-----------+
           |  LambdaRank reranker  |  LightGBM, trained out-of-fold,
           |                       |  shared across both systems
           +-----------+-----------+
                       |
                top 10 candidates
                       |
           +-----------v-----------+
           |  Cross-encoder        |  MedCPT, fine-tuned on the coding team's
           |                       |  decisions, blended with the ranker
           +-----------+-----------+
                       |
          code + confidence + cross-system check
```

**Normalisation.** Terms are lowercased and stripped of markup and punctuation. Abbreviations are expanded from
a curated table into one or more query variants. Each reading of an ambiguous abbreviation carries context cue
words (right atrium: *echo, tricuspid, rv*; rheumatoid arthritis: *methotrexate, synovitis, stiffness*). When the
chunk text supports exactly one reading, the term is searched under that reading; otherwise every reading is kept
and the ranker decides.

**Retrieval.** Each query variant is searched against every description of the target system:

| Source | What it catches |
|---|---|
| Character 3-4 gram TF-IDF | misspellings, run-together words, truncation |
| Word 1-2 gram TF-IDF | exact rare tokens, eponyms, multi-word names |
| Training-term aliases | local coding convention and shorthand absent from any dictionary |
| SapBERT embeddings | synonyms with no lexical overlap (`heart attack` / `myocardial infarction`) |
| BioLORD-2023 embeddings | definition-grounded similarity; complements SapBERT |

For ICD-10-CM, a retrieved non-billable category also contributes its *unspecified* or *without complication*
children, so the ranker can choose the default code when the text supports nothing more specific.

**Reranking.** Every candidate is described by 65 features:

- retrieval score, gap to the best score and rank, from each source
- surface similarity to the matched description and to the preferred term
- token coverage in both directions, and the share of *unexplained* description tokens that the chunk context
  accounts for (the main signal for specificity calibration)
- qualifier agreement: laterality, acuity, history-of, family history, negation, type, stage, encounter
- coding-team priors: frequency of the code and of its ICD block or SNOMED semantic tag among gold labels
- ICD structure: leaf status, depth, unspecified and other flags
- subconcept agreement, when the input carries a `subconcept` column: P(SNOMED tag or ICD block | subconcept) and
  P(code | subconcept)

A LightGBM `LGBMRanker` (LambdaRank objective) orders the candidates of each term. One ranker is shared by both
systems, with a system indicator feature, so ICD-10-CM benefits from the larger SNOMED CT sample. Two variants are
trained, with and without the subconcept features, and the one matching the input schema is used.

**Cross-encoder.** `ncbi/MedCPT-Cross-Encoder`, fine-tuned on the coding team's decisions, re-scores the top 10
candidates. It reads the term, its subconcept and a context window together with each candidate description.
Training groups pair the gold description with 15 hard negatives taken from the ranker's own out-of-fold candidate
lists, with a listwise softmax loss. Final scores are a log-linear blend:
`log p_ranker + 0.5 * log p_cross_encoder`.

**Cross-system check.** 2,124 labelled terms carry both an ICD-10-CM and a SNOMED CT gold code. From them the engine
learns which code pairs the coding team uses, and reports whether its two independent predictions form such a pair.
On held-out data, ICD-10-CM predictions are 88.9% accurate when the pair agrees and 31.8% when the team has paired
these codes differently. That makes it a strong signal for routing terms to review.

**Leakage control.** The alias index and label priors are built from training data, so the ranker is trained on
out-of-fold candidates. Within 5 folds, each fold's features come from an alias index and priors built on the
other four. The cross-encoder's training groups are built the same way.

---

## Results

Held-out evaluation: 80/20 random split of the labelled data, seed 13. The 20% is never seen by retrieval, priors,
the ranker or the cross-encoder. Test terms: 472 ICD-10-CM, 1,529 SNOMED CT. In this split, 24% of ICD-10-CM and
47% of SNOMED CT test terms have a gold code that never occurs in the training portion.

**Stage by stage** (exact-match accuracy, %)

| Configuration | ICD-10-CM | SNOMED CT |
|---|---|---|
| Lexical retrieval + aliases, gradient boosting per system | 54.0 | 41.7 |
| + shared ranker, subconcept priors | 53.4 | 42.4 |
| + LambdaRank objective | 54.5 | 43.0 |
| + SapBERT dense retrieval | 61.7 | 49.1 |
| + BioLORD-2023 dense retrieval | 65.0 | 49.6 |
| **+ fine-tuned MedCPT cross-encoder** | **68.9** | **53.8** |

**Headline (lower of the two): 53.8**

Blend weights were chosen with cross-fitting: selected on one half of the test terms, scored on the other half.
The cross-fitted accuracy equals the reported accuracy for both systems.

Context cues for ambiguous abbreviations (`data/resources/abbreviations.tsv`, third column) were added after the
table above. On the same split they leave ICD-10-CM at 68.9 with the cross-encoder, and move the SNOMED CT ranker
stage from 49.6 to 49.7. Bare abbreviations whose reading the context settles (`ra` next to methotrexate vs. in an
echo report) are searched under that reading only.

**Diagnostics**

| Metric | ICD-10-CM | SNOMED CT |
|---|---|---|
| Gold anywhere in the candidate set | 90.3 | 86.7 |
| Gold in the top 5 / top 10 | 83.9 / 87.1 | 73.1 / 78.6 |
| Accuracy, gold code seen in training | 76.3 | 64.1 |
| Accuracy, gold code unseen in training (long tail) | 45.1 | 42.2 |
| ICD 3-character block accuracy (right category, any specificity) | 81.1 | |
| Cross-encoder alone | 59.1 | 48.7 |
| Ranker without the subconcept column | 64.8 | 49.1 |

**Cross-system agreement as confidence** (ICD-10-CM targets)

| Predicted ICD/SNOMED pair | Share of terms | Accuracy |
|---|---|---|
| used by the coding team | 36.2 | 88.9 |
| codes paired by the team, but differently | 9.3 | 31.8 |
| no pairing evidence | 54.5 | 61.9 |

**Other variants measured**

| Variant | ICD-10-CM | SNOMED CT | Outcome |
|---|---|---|---|
| Zero-shot `Qwen2.5-7B-Instruct` adjudication of the top 10 (at the SapBERT stage, cross-fitted) | 58.7 | 53.4 | helps SNOMED CT, hurts ICD-10-CM; replaced by the cross-encoder |
| Crosswalk prior blended into scores | 68.4 | 53.8 | no robust gain under cross-fitting; kept as a confidence signal only |
| Gold display texts as extra aliases | 64.0 | 49.5 | lowered accuracy; removed |

The failure-mode breakdown is produced by `scripts/error_analysis.py` and discussed in the documentation.

Reproduce with `python -m termmap evaluate --tag <name>`. Per-term outputs, the report and the second-stage request
and training files are written to `outputs/eval_<name>/`.

---

## Setup

Python 3.10 or later.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# dense retrieval and cross-encoder scoring on the local machine
pip install -r requirements-dense.txt
```

Place the challenge files in `data/raw/`:

```
data/raw/hackathon_train.csv
data/raw/icd10cm_codes.xlsx
data/raw/SNOMEDCT_US_codes_and_descriptions.xlsx
```

### Trained artifacts

The trained engine, the fine-tuned cross-encoder and the description embeddings are derived from the challenge
data, so they are not stored in this repository. They are kept in a **private** Hugging Face Hub repository.
With read access to it, set `HF_TOKEN` and `TERMMAP_ARTIFACTS_REPO` in `.env` (see `.env.example`) and run:

```bash
python -m termmap fetch-artifacts                  # about 2.8 GB, into artifacts/
python -m termmap fetch-artifacts --no-embeddings  # about 0.5 GB, skips the description embeddings
```

| File in `artifacts/` | Content |
|---|---|
| `engine.joblib` | ranker, alias index, priors, coding-team crosswalk |
| `ce_model/` | fine-tuned MedCPT cross-encoder |
| `terminology_*.parquet` | parsed dictionaries |
| `dense_hits_*.parquet` | precomputed dense hits for the labelled terms |
| `dense_*.emb.npy`, `dense_*.codes.npy` | SapBERT and BioLORD description embeddings, used to encode new terms |

Every artifact can also be rebuilt from the challenge files: `python -m termmap train` rebuilds the engine, and the
notebooks in [GPU steps](#gpu-steps) rebuild the embeddings and the cross-encoder. Without the cross-encoder and
embeddings the engine still runs end to end, on lexical retrieval and the ranker.

### Environment variables

Copy `.env.example` to `.env` and fill in the values. `.env` is listed in `.gitignore`; keys are only read from
the environment and are never written to disk by the code.

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token: read access to the artifacts repository, and model downloads |
| `TERMMAP_ARTIFACTS_REPO` | Hub repository id holding the trained artifacts |
| `TERMMAP_FORCE_CPU` | set to `1` to keep encoders on CPU when a GPU is present |
| `TERMMAP_CPU_THREADS` | threads used by encoders on CPU (default 2) |

---

## Running the engine

**Reproduce the submitted predictions (single command, after `fetch-artifacts`):**

```bash
python -m termmap predict --input data/raw/<evaluation_file>.csv --output outputs/predictions.csv
```

Without fetched artifacts, the first run parses the dictionaries, trains the engine on `data/raw/hackathon_train.csv` and caches everything
under `artifacts/`. Later runs reuse the cache. The input must contain an `entity` or `clinical_entity` column.
`chunk_text` is used as context and `subconcept` as a prior when present, and `chunk_id` is carried through. New terms are
encoded on the fly when the description embeddings are present, and scored by the cross-encoder when
`artifacts/ce_model/` is present. The output contains `chunk_id, clinical_entity, icd10cm_code, snomed_code` for every
row. Add `--with-confidence` to include per-system confidence scores.

**Other commands:**

```bash
python -m termmap train                       # fit and save the engine, write fine-tuning data
python -m termmap evaluate --tag final        # held-out evaluation, diagnostics and error files
python -m termmap map "afib w rvr" --context "ekg shows irregularly irregular rhythm"
python scripts/error_analysis.py outputs/eval_final
```

---

## Demo app

```bash
streamlit run app.py
```

Maps a term with optional context and entity category, shows the top candidates and confidence for both systems and
the cross-system check, and maps a whole CSV file with a download of the predictions. The examples above the input
box cover an OCR-damaged term, an abbreviation collision (`ms` in a cardiology note and in a neurology note) and a
specificity decision.

---

## GPU steps

Encoders and the cross-encoder fine-tune were run on a single T4 GPU. Each notebook reads and writes one Google Drive
folder and documents its inputs and outputs.

| Notebook | Inputs | Outputs |
|---|---|---|
| `notebooks/build_dense_index.ipynb` | `artifacts/terminology_*.desc.parquet`, query list from `python -m termmap export-queries` | dense hits per query, cached description embeddings |
| `notebooks/cross_encoder.ipynb` | `ce_train_*.jsonl`, `ce_requests_*.jsonl` from `evaluate` or `train` / `export-stage2` | fine-tuned model, `ce_scores_*.jsonl` |
| `notebooks/llm_adjudication.ipynb` | `llm_requests_*.jsonl` | `llm_scores_*.jsonl` (zero-shot) |
| `notebooks/lora_selector.ipynb` | `llm_train_*.jsonl`, `llm_requests_*.jsonl` | LoRA adapter, `lora_scores_*.jsonl` |

Score files placed in `artifacts/` are picked up automatically by request id. Blend weights per system are set in
`termmap/config.py` (`stage2_weights`).

---

## Configuration

Defaults live in `termmap/config.py` and can be overridden with a YAML file:

```bash
python -m termmap --config configs/default.yaml evaluate
```

| Key | Default | Meaning |
|---|---|---|
| `k_char`, `k_word`, `k_alias`, `k_dense` | 40, 40, 20, 40 | codes kept per source per query variant |
| `max_candidates` | 60 | candidate set size passed to the ranker |
| `n_folds` | 5 | folds for out-of-fold training |
| `seed` | 13 | fixed seed for every split and model |
| `stage2_top_n` | 10 | candidates re-scored by the second stage |
| `stage2_weights` | `ce: 0.5` for both systems | log-linear blend weights |
| `coupling_weights` | 0 | crosswalk prior weight (diagnostic only) |

The pipeline is deterministic. All randomness comes from `seed`, LightGBM runs in deterministic mode, and there is no
sampling at inference.

---

## Repository layout

```
termmap/
  normalize.py     text cleaning, abbreviation expansion
  resources.py     dictionary loading, semantic tags, ICD hierarchy, input parsing
  index.py         sparse TF-IDF index
  candidates.py    retrieval sources and candidate merging
  dense.py         dense retrieval (precomputed hits or live encoding)
  features.py      candidate feature extraction
  engine.py        per-system mapper, out-of-fold training, ranker, second stage, persistence
  stage2.py        cross-encoder and LLM requests, fine-tuning data, score blending
  coupling.py      coding-team crosswalk between ICD-10-CM and SNOMED CT
  evaluate.py      held-out evaluation and diagnostics
  hub.py           fetching artifacts from the private Hub repository
  __main__.py      command line interface
app.py             Streamlit demo
scripts/error_analysis.py
scripts/upload_artifacts.py
data/resources/abbreviations.tsv
configs/default.yaml
notebooks/         GPU notebooks
docs/              documentation (PDF and HTML source)
```

---

## Design notes

**Why rerank rather than threshold a similarity score.** Lexical and clinical distance are poorly correlated exactly
where errors happen. `hypertension` is one token away from `pulmonary hypertension`. A learned ranker can weigh an
unexplained token like `pulmonary` against the context, the priors and the other sources.

**Why predict both systems independently.** ICD-10-CM and SNOMED CT differ in granularity. Deriving one from the
other imports the crosswalk's gaps. Each system has its own retrieval, priors and ranking. The coding-team crosswalk
is only used to check agreement after both predictions are made.

**Specificity calibration.** Two features carry most of this: unexplained description tokens, and the share of those
tokens supported by the chunk context. Together with default-child expansion for ICD-10-CM, the model can choose
`E11.9` when the text says only "diabetes" and a specific code when the context names the complication.

**Version drift.** All terminology knowledge is read from the supplied dictionary files at build time. Nothing is
hardcoded. A new ICD-10-CM or SNOMED CT release is picked up by replacing the files and clearing `artifacts/`.

**Throughput.** On a 4-core laptop CPU (2 threads), the lexical pipeline and ranker map a term to both systems in
about 28 ms once the indexes are built. Index construction takes about one minute at start-up. Encoding new queries and
cross-encoder scoring add a BERT-base forward pass per query and per candidate, which a GPU absorbs at chart volume.

---

## External resources

| Resource | Use | Licence |
|---|---|---|
| ICD-10-CM code descriptions (supplied) | answer space, retrieval | public domain (CDC/NCHS) |
| SNOMED CT US Edition descriptions (supplied, export of 24 Sep 2026) | answer space, retrieval | SNOMED CT affiliate licence |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | dense retrieval | Apache 2.0 |
| `FremyCompany/BioLORD-2023` | dense retrieval | see model card |
| `ncbi/MedCPT-Cross-Encoder` | cross-encoder, fine-tuned | see model card |
| `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` | LLM experiments | Apache 2.0 / Qwen licence |
| LightGBM, scikit-learn, RapidFuzz, pandas, PyTorch, Transformers, PEFT, Streamlit | modelling, demo | MIT / BSD / Apache 2.0 |

No official crosswalk or UMLS release was used.
