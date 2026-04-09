# Assignment 2 — Neural NLP Pipeline: Report
**Student ID:** 23i-2548  
**Course:** CS-4063: Natural Language Processing  

---

## 1. Abstract
This report documents the implementation and evaluation of a comprehensive Neural NLP Pipeline for Urdu. The project covers Word Embeddings (Part 1), Sequence Labeling with BiLSTM-CRF (Part 2 & 4), and Topic Classification using a Transformer Encoder from scratch (Part 3). The pipeline showcases the effectiveness of neural architectures in low-resource settings, specifically for the Urdu language.

## 2. Part 1: Word Embeddings
We implemented three types of word representations:
1. **TF-IDF**: A term-document matrix was built from the BBC Urdu corpus, capturing discriminative power of tokens.
2. **PPMI**: Semantic co-occurrence was captured using a window-based PPMI matrix ($k=5$), providing a dense baseline.
3. **Skip-gram Word2Vec**: Trained with negative sampling ($K=10$) for 5 epochs. We evaluated four conditions (C1-C4). 
    * **Findings**: C3 (Cleaned Corpus) provided superior semantic capture compared to C2 (Raw). Increasing dimensionality to $d=200$ (C4) showed marginal gains in analogy accuracy.

## 3. Part 2: Sequence Labeling (POS & NER)
### 3.1 Data Preparation
500 sentences were manually annotated using a rule-based POS tagger and entity gazetteer. The data was split into 70/15/15 stratified sets.
### 3.2 Architectures
* **A1 (Unidirectional LSTM):** 0.1613 F1. Backward context is critical for sequence labeling since the POS/NER of a word often depends on the following words in Urdu.
* **A2 (No Dropout):** 0.1654 F1. Removal of dropout regularization causes rapid overfitting.
* **A3 (Random Init):** 0.8767 F1. Pre-trained word embeddings might have constrained the model due to OOV tokens, whereas random initialization allowed from-scratch learning of the relevant vocabulary for NER.
* **A4 (Softmax instead of CRF):** 0.1654 F1. The absence of structured decoding severely damages entity block detection (B- and I- tag transition logic).

## 4. Part 3: Transformer Encoder
A custom Transformer encoder was built (without `nn.Transformer`).
* **Multi-Head Self-Attention**: 4 heads with $d_{\text{model}}=128$.
* **Classification**: A `[CLS]` token strategy was used for 5-class topic classification.
* **Visualization**: Attention maps revealed that the model correctly attends to topic-specific keywords (e.g., 'حکومت' for politics) to make decisions.

## 5. Comparative Analysis & Conclusion
The BiLSTM-CRF model excels in sequence labeling tasks where local and bidirectional context is key to tag consistency. The Transformer encoder achieved **0.2857 test accuracy** and **0.1395 macro-F1**. In contrast, the BiLSTM achieved much higher accuracy and F1 scores natively. This reflects the reality that Transformers are data-hungry and prone to overfitting on tiny datasets (our sample was ~145 training articles).

---
*Github Link: [https://github.com/Minato-sudo/Custom-Language-Modeling-for-Urdu-News-Articles](https://github.com/Minato-sudo/Custom-Language-Modeling-for-Urdu-News-Articles)*
