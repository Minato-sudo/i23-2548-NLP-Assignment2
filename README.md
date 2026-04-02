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
| 2 | Day 1 (Apr 3) | Vocabulary + TF-IDF + PPMI |
| 3 | Day 1 (Apr 3) | Word2Vec Skip-Gram + Analogy Tests |
| 4 | Day 2 (Apr 4) | Annotation + POS Tagger + NER |
| 5 | Day 3 (Apr 5) | Transformer Encoder + Classifier |
| 6 | Day 4 (Apr 6) | CRF + Viterbi + Final Polish |

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
