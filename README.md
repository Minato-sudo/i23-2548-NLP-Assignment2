# CS-4063: Natural Language Processing - Assignment 2  
**Neural NLP Pipeline for Urdu | PyTorch from Scratch**

**Student ID:** 23i-2548  
**Semester:** Spring 2025  
**GitHub:** [Minato-sudo/i23-2548-NLP-Assignment2](https://github.com/Minato-sudo/i23-2548-NLP-Assignment2)

---

## Overview
This repository contains a complete **Neural NLP Pipeline for Urdu** implemented entirely from scratch in PyTorch (no HuggingFace, no Gensim, and no forbidden built-ins: `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder`).

The pipeline includes:
- **Part 1**: TF-IDF, PPMI, and Skip-gram Word2Vec embeddings
- **Part 2**: 2-layer BiLSTM-CRF for POS tagging and NER
- **Part 3**: Custom Transformer Encoder for 5-class topic classification

All components were trained and evaluated on the **BBC Urdu News** corpus (`cleaned.txt`).

---

## Folder Structure
i23-2548-NLP-Assignment2/
├── data/
│   ├── pos_train.conll
│   ├── pos_test.conll
│   ├── ner_train.conll
│   └── ner_test.conll
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
├── report.pdf
├── 23i-2548_Assignment2_DS-X.ipynb
├── README.md
└── src/                  # (optional) Python scripts used
text---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- numpy, pandas, scikit-learn, matplotlib, seaborn
- `seqeval` (for NER evaluation)
- XeLaTeX (only if you want to recompile the report)

---

## How to Reproduce

### 1. Environment Setup
```bash
# Clone the repo
git clone https://github.com/Minato-sudo/i23-2548-NLP-Assignment2.git
cd i23-2548-NLP-Assignment2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn seqeval
2. Reproduce Each Part
Part 1 – Word Embeddings
Bash# Run the notebook cells for Part 1
# Or run the script:
python src/part1_embeddings.py
→ Generates embeddings/ folder (tfidf_matrix.npy, ppmi_matrix.npy, embeddings_w2v.npy)
Part 2 – BiLSTM-CRF (POS + NER)
Bash# Run the notebook cells for Part 2
# Or run:
python src/part2_bilstm_crf.py
→ Generates models/bilstm_pos.pt and models/bilstm_ner.pt + data/ CONLL files
Part 3 – Transformer Encoder
Bash# Run the notebook cells for Part 3
# Or run:
python src/part3_transformer.py
→ Generates models/transformer_cls.pt
3. View Results

Open 23i-2548_Assignment2_DS-X.ipynb (all cells are already executed)
Read the final report: report.pdf


Report
The complete assignment report (report.pdf) is included in the root directory.
It contains all results, tables, figures, ablation studies, and comparative analysis as per the assignment rubric.

GitHub Submission Notes

Incremental commit history: ≥5 meaningful commits (no single bulk commit)
All code is fully reproducible
Report satisfies the 2–3 page requirement (Times New Roman 12pt, 1.5 spacing)


Any questions? Feel free to open an issue or contact me.
Submitted by:
23i-2548
Spring 2025
text---

### How to use it:
1. Go to your GitHub repo: https://github.com/Minato-sudo/i23-2548-NLP-Assignment2
2. Click on **"Add file" → "Create new file"**
3. Name the file: `README.md`
4. Paste the entire content above
5. Click **"Commit new file"**

Done!