# Medical Specialty Classification Using BioClinicalBERT

## Overview

**Medical Specialty Classification Using BioClinicalBERT** is a production-grade clinical NLP project that classifies real-world medical transcription notes into their corresponding medical specialties using a fine-tuned domain-specific transformer. Built on `emilyalsentzer/Bio_ClinicalBERT` — a BERT model pretrained on MIMIC-III clinical notes from the Beth Israel Deaconess Medical Center — the system applies structured multi-field input construction, overlapping chunk-based long-document handling, class-weighted fine-tuning, and note-level logit aggregation to deliver deployment-realistic specialty prediction across a diverse set of clinical categories.

Every design decision in this project reflects the constraints and priorities of real clinical NLP work: domain-adapted language modeling, preservation of long-document context, handling of severe label imbalance, and evaluation protocols that mirror how the model would actually behave in production.

---

## Why This Project Matters

Automatic medical specialty routing from transcription notes is a high-value problem in clinical workflow automation, medical coding, and EHR triage systems. This project demonstrates how modern clinical NLP techniques — domain-specific pretraining, long-document chunking, and imbalance-aware fine-tuning — can be combined into a rigorous, deployment-ready classification pipeline built entirely on open-source tools and real clinical text.

---

## Key Highlights

- Fine-tuned `emilyalsentzer/Bio_ClinicalBERT` on the MTSamples real-world medical transcription dataset
- Structured multi-field clinical input: sample name, description, keywords, and transcription joined with `[SEP]` delimiters for richer contextual signal
- Overlapping sliding-window chunk strategy (stride = 96 tokens) to handle notes exceeding 512 tokens without hard truncation or information loss
- Note-level inference via chunk logit averaging — one final prediction per full clinical document, mirroring true deployment behavior
- Two-phase training: frozen-encoder head warm-up followed by full end-to-end fine-tuning
- Custom `WeightedTrainer` subclass applying class-weighted cross-entropy loss with weights clipped to `[1.0, 5.0]` for imbalance-robust optimization
- Cosine annealing learning rate schedule with linear warmup and early stopping on macro F1
- Mixed precision training (FP16) for GPU efficiency
- Minority-class text augmentation module via `nlpaug` (configurable, disabled by default to preserve clinical terminology)
- Dual-granularity evaluation: chunk-level and note-level metrics, confusion matrices, and per-class accuracy analysis
- Confidence distribution profiling and systematic error analysis on misclassified notes
- Final predictions exported as `note_level_test_predictions.csv` for downstream review

---

## Repository Contents

| File | Description |
|---|---|
| `medical_specialty_classification_using_bioclinicalbert.ipynb` | Complete end-to-end notebook |
| `mtsamples.zip` | MTSamples dataset (included — no external download needed) |
| `note_level_test_predictions.csv` | Exported note-level test set predictions |
| `README.md` | Project documentation |
| `LICENSE` | MIT License |

---

## Dataset

This project uses the **MTSamples** dataset — a publicly available collection of real de-identified medical transcription notes across a broad range of clinical specialties. The dataset is included in this repository as `mtsamples.zip` and requires no external download.

### Input Fields Used

| Field | Role |
|---|---|
| `medical_specialty` | Target label |
| `sample_name` | Structured input component |
| `description` | Structured input component |
| `keywords` | Structured input component |
| `transcription` | Primary clinical text |

### Specialty Categories

`Surgery`, `Consult - History and Phy.`, `Cardiovascular / Pulmonary`, `Orthopedic`, `Radiology`, `General Medicine`, `Gastroenterology`, `Neurology`, `SOAP / Chart / Progress Notes`, `Urology`, and more. Classes with fewer than 80 samples are filtered before training to ensure stable gradient estimation.

---

## System Architecture

### Pipeline Overview

1. Extract and load `mtsamples.csv` from `mtsamples.zip`
2. Clean text fields, drop nulls, strip whitespace, remove duplicate records
3. Construct structured multi-field input per note: `"Sample name: ... [SEP] Description: ... [SEP] Keywords: ... [SEP] Transcription: ..."`
4. Filter low-sample specialties (minimum threshold: 80 samples)
5. Exploratory data analysis: class distribution, input length distribution, 90th percentile token count
6. Label encoding and stratified train / validation / test split (80 / 10 / 10)
7. Optional minority-class synonym augmentation via `nlpaug` (disabled by default)
8. Compute balanced class weights, clipped to `[1.0, 5.0]`
9. Tokenize with BioClinicalBERT tokenizer at `max_length=384`
10. Expand notes into overlapping chunks (stride = 96 tokens, window = 382 tokens)
11. Build HuggingFace `Dataset` objects with `DataCollatorWithPadding`
12. Phase 1 — Warm-Up: train classification head only for 1 epoch (`lr=5e-4`, frozen encoder)
13. Phase 2 — Full Fine-Tuning: all layers, cosine LR, linear warmup, early stopping on macro F1
14. Chunk-level evaluation: accuracy, macro F1, weighted F1, confusion matrix, per-class accuracy
15. Note-level aggregation: average chunk logits per note, argmax for final label
16. Note-level evaluation: accuracy, macro F1, weighted F1, confusion matrix, per-class accuracy
17. Confidence distribution analysis and error analysis on misclassified notes
18. Export `note_level_test_predictions.csv`

---

## Machine Learning Methodology

### Structured Input Construction

All five available clinical metadata fields are concatenated into a single structured string per note using `[SEP]` as a field boundary marker. This provides the model with specialty-relevant signals from keywords, descriptive summaries, and sample names alongside the full transcription — significantly more context than transcription text alone.

### Long-Document Handling

Clinical notes routinely exceed the 512-token context limit of BERT-family models. Rather than truncating, the pipeline applies an overlapping sliding window:

- Each note is fully tokenized without truncation
- Token windows of 382 tokens are extracted with a stride of 96 tokens, producing overlapping chunks that share context at boundaries
- Each window is decoded back to text and re-tokenized with special tokens added
- At inference time, logits from all chunks of a note are averaged before the final argmax — one prediction per full document

This strategy preserves information from the complete clinical record while remaining compatible with fixed-context transformer architectures.

### Two-Phase Training

**Phase 1 — Head Warm-Up:** The BioClinicalBERT encoder is fully frozen. Only the classification head is trained for one epoch at `lr=5e-4`. This prevents the pretrained clinical representations from being corrupted by noisy early gradients before the head has learned task-relevant features.

**Phase 2 — Full Fine-Tuning:** All encoder layers are unfrozen. End-to-end fine-tuning proceeds at `lr=2e-5` with cosine annealing, linear warmup over 10% of total steps, and early stopping based on validation macro F1 with patience of 2 epochs. The best checkpoint is automatically restored at the end of training.

### Class Imbalance Strategy

MTSamples is heavily skewed across specialties. Imbalance is addressed at multiple levels:

- **Balanced class weights** computed via `compute_class_weight("balanced")` and clipped to `[1.0, 5.0]` to prevent gradient instability from extreme weights
- **Custom `WeightedTrainer`** subclassing HuggingFace `Trainer` to inject weighted cross-entropy at every optimization step
- **Low-sample class filtering** removing any specialty with fewer than 80 samples before training begins
- **Optional text augmentation** for minority classes using WordNet synonym replacement via `nlpaug` (disabled by default to preserve clinical terminology integrity)

### Training Configuration

| Parameter | Phase 1 — Warm-Up | Phase 2 — Fine-Tuning |
|---|---|---|
| Backbone | Frozen | Unfrozen |
| Learning Rate | 5e-4 | 2e-5 |
| LR Scheduler | — | Cosine Annealing |
| Warmup Steps | — | 10% of total steps |
| Epochs | 1 | Up to 6 (early stopping) |
| Early Stopping Patience | — | 2 epochs |
| Best Model Metric | — | Macro F1 |
| Per-Device Batch Size | 4 | 4 |
| Gradient Accumulation | 4 steps | 4 steps |
| Effective Batch Size | 16 | 16 |
| Mixed Precision | FP16 | FP16 |
| Loss Function | Weighted CrossEntropy | Weighted CrossEntropy |
| Max Sequence Length | 384 tokens | 384 tokens |
| Chunk Stride | 96 tokens | 96 tokens |

---

## Evaluation

The model is evaluated at two distinct granularities, each serving a different diagnostic purpose.

### Chunk-Level Evaluation

Every individual chunk of every test note is classified independently. This produces a large evaluation set and reveals how well the model handles partial clinical document context.

### Note-Level Evaluation

Chunk logits for each note are averaged and a single argmax prediction is produced per document. This is the primary evaluation — it reflects exactly how the model would behave in a real deployment where one prediction is required per patient note.

### Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct prediction rate |
| Macro F1 | Unweighted average F1 — primary model selection metric |
| Weighted F1 | F1 weighted by class support |
| Per-Class Accuracy | Specialty-level breakdown sorted highest to lowest |
| Confusion Matrix | Normalized heatmap with raw counts for class interaction analysis |
| Confidence Distribution | Histogram of maximum predicted probabilities with mean, median, and 90th percentile |
| Error Analysis | Tabular review of misclassified notes with true and predicted specialty labels |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/medical-specialty-classification-bioclinicalbert.git
cd medical-specialty-classification-bioclinicalbert
```

### Install Dependencies

```bash
pip install transformers datasets accelerate scikit-learn pandas numpy matplotlib seaborn torch sentencepiece
```

---

## Usage

1. Open `medical_specialty_classification_using_bioclinicalbert.ipynb` in Google Colab or Jupyter Notebook.
2. `mtsamples.zip` is already included in the repository — no external data download required.
3. Run all notebook cells sequentially. The pipeline is self-contained and fully reproducible with a fixed global seed across Python, NumPy, PyTorch, and HuggingFace.
4. All evaluation outputs render inline. The final prediction file is saved as `note_level_test_predictions.csv`.

---

## Technology Stack

- Python
- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Colab

---

## Engineering Principles

- Domain-adapted pretraining via BioClinicalBERT for clinical language understanding out of the box
- Multi-field structured input construction to maximize available clinical context per note
- Overlapping chunk strategy with note-level logit averaging to handle long documents without truncation loss
- Two-phase training isolating head warm-up from full encoder fine-tuning for stable optimization
- Clipped class-weighted loss providing imbalance correction without gradient instability
- Cosine LR schedule with warmup preventing early training divergence
- Early stopping on macro F1 selecting the best generalizing checkpoint automatically
- Dual-granularity evaluation distinguishing chunk-level training performance from note-level deployment performance
- Confidence distribution analysis assessing calibration alongside standard accuracy metrics
- Error analysis surfacing systematic misclassification patterns for targeted improvement
- Fully reproducible pipeline with a fixed seed propagated across all stochastic components

---

## Potential Extensions

- Longformer or BigBird backbone to natively handle full clinical note context beyond 512 tokens
- Ensemble of BioClinicalBERT, ClinicalBERT, and PubMedBERT variants for improved specialty coverage
- Multi-label classification for notes that span multiple clinical specialties simultaneously
- Named entity recognition integrated alongside classification for joint clinical information extraction
- SHAP or attention-weight visualization for clinical interpretability and model validation
- Flask or FastAPI REST API wrapping the note-level inference function for real-time deployment
- ONNX export for cross-platform inference and edge deployment
- TorchScript conversion for mobile and embedded clinical device deployment

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**  
[GitHub](https://github.com/sushantkothari)
