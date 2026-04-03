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
| 1 | Day 0 (Apr 2) | Repo scaffolding & README |
| 2 | Day 1 (Apr 3) | Vocabulary + TF-IDF + PPMI | [DONE] |
| 3 | Day 1 (Apr 3) | Word2Vec Skip-Gram + Analogy Tests | [DONE] |
| 4 | Day 2 (Apr 4) | Annotation + POS Tagger + NER | [TODO] |
| 5 | Day 3 (Apr 5) | Transformer Encoder + Classifier | [TODO] |
| 6 | Day 4 (Apr 6) | CRF + Viterbi + Final Polish | [TODO] |

---

## Part 1: Summary of Results

### 1.1 Matrices
- **Vocabulary**: 10,000 tokens + `<UNK>` from BBC Urdu corpus.
- **TF-IDF**: Captures discriminative words for specific documents.
- **PPMI**: Captures semantic co-occurrence with a window of $k=5$.

### 1.2 Skip-gram Word2Vec
- **Architecture**: Separate $V$ and $U$ matrices, $d=100/200$.
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

## Requirements

```
torch
numpy
scikit-learn
matplotlib
```

---

*Work in progress — updated daily.*
**Last Update:** Apr 3 (Evening) - Part 1 completely verified and documented.