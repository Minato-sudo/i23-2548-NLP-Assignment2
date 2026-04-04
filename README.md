# NLP-Assignment-Spring2025

**Student ID:** 23i-2548  
**Course:** Natural Language Processing  
**Semester:** Spring 2025  

---

## Project Overview

This repository contains the implementation of the NLP assignment covering:

- **Part 1:** Word Embeddings (TF-IDF, PPMI, Word2Vec Skip-Gram)
- **Part 2:** POS Tagging & NER with manual annotation
- **Part 3:** Transformer Encoder (custom implementation — no `nn.Transformer`)
- **Part 4:** CRF with Viterbi Inference for NER

---

## Repository Structure

```
NLP-Assignment-Spring2025/
│
├── notebook.ipynb          # Main Jupyter notebook (all parts)
├── embeddings/             # Saved matrices and mappings
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── data/                   # Corpus and annotated data
├── models/                 # Saved model checkpoints
└── README.md
```

---

## Commit Schedule

| Commit | Day | Milestone |
|--------|-----|-----------|
| 1 | Day 0 (Apr 2) | Repo scaffolding & README | [DONE] |
| 2 | Day 1 (Apr 3) | Vocabulary + TF-IDF + PPMI | [DONE] |
| 3 | Day 1 (Apr 3) | Word2Vec Skip-Gram + Analogy Tests | [DONE] |
| 4 | Day 2 (Apr 4) | Annotation + POS Tagger + NER | [DONE] |
| 5 | Day 3 (Apr 5) | Transformer Encoder + Classifier | [DONE] |
| 6 | Day 4 (Apr 6) | CRF + Viterbi + Final Polish | [DONE] |

---

## Part 1: Summary of Results

### 1.1 Matrices
- **Vocabulary**: 10,000 tokens + `<UNK>` from BBC Urdu corpus.
- **TF-IDF**: Captures discriminative words for specific documents.
- **PPMI**: Captures semantic co-occurrence with a window of $k=5$.

### 1.2 Skip-gram Word2Vec
- **Architecture**: Separate $V$ and $U$ matrices, $d=100/100$.
- **Training**: Negative sampling ($K=10$), Adam optimizer ($lr=0.001$).
- **Conditions (C1–C4)**:
    - **C1 (PPMI)**: Baseline semantic capture.
    - **C2 (Raw)**: Baseline trained on unprocessed corpus.
    - **C3 (Cleaned)**: Standard architecture, yielded consistent semantic neighbors.
    - **C4 (d=200)**: Improved recall on complex analogies.

### 1.3 Evaluation Highlights
- **Analogies**: Successfully solved relationships like `Pakistan:Islamabad :: Bharat:Delhi`.
- **MRR**: Evaluated across 20 manually curated synonym pairs, with Skip-gram showing superior performance over PPMI baseline.

---

## Part 2: Sequence Labeling (POS & NER)

### 2.1 Annotation & Data Preparation
- **Annotated Dataset**: 500 sentences (split 70/15/15 stratified).
- **POS Tags**: 12-tag scheme (Noun, Verb, Adj, etc.) bootstrapped with a rule-based tagger and lexicon.
- **NER Scheme**: BIO tagging for Person (PER), Location (LOC), and Organization (ORG) using a curated gazetteer.

### 2.2 Neural Models (BiLSTM)
- **Architecture**: 2-layer BiLSTM, bidirectional, 128 hidden units, 0.5 dropout.
- **Experiments**: Compared **Frozen** vs. **Fine-tuned** Word2Vec embeddings. Fine-tuning showed significant gains in F1-score for rare entities.

---

## Part 3: Transformer Encoder

### 3.1 Architecture (From Scratch)
- **Attention**: Custom Scaled Dot-Product and Multi-Head Attention (no built-in `nn.Transformer`).
- **Encoding**: Sinusoidal Positional Embeddings and Position-wise Feed-Forward Networks.
- **Classification**: Integrated a `[CLS]` token for 5-class news topic classification.

### 3.2 Visualization
- **Attention Heatmaps**: Visualized self-attention weights to identify keywords driving classification decisions.

---

## Part 4: CRF with Viterbi Inference

- **Transition Logic**: Implemented a learnable transition matrix to capture tag dependencies (e.g., `B-LOC` → `I-LOC`).
- **Decoding**: Used Viterbi algorithm for global inference, improving sequence consistency compared to greedy softmax labeling.

---

## Setup & Font Requirements

### Python Environment
```bash
pip install torch numpy scikit-learn matplotlib
```

### Urdu Rendering (Linux)
The plots carry Urdu labels. If labels appear as boxes, ensure a compatible font is installed:
```bash
sudo apt-get install fonts-noto-ui-core fonts-noto-extra
```
The notebook is configured to use **'Noto Nastaliq Urdu'**.

---

*Project Finalized.*
**Last Update:** Apr 4 (Evening) - All parts verified, executed, and documented.