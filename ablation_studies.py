import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from models.sequence_labeler import BiLSTMModel, BiLSTM_CRF
from train_part2 import load_conll, SequenceDataset, collate_fn, train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_ablation(a_id, task_type="ner"):
    print(f"\n--- Running Ablation {a_id} ({task_type.upper()}) ---")
    
    # Defaults
    bidirectional = True
    dropout = 0.5
    frozen = False # Fine-tuned
    use_crf = (task_type == "ner")
    use_pretrained = True
    
    if a_id == "A1":
        bidirectional = False
    elif a_id == "A2":
        dropout = 0.0
    elif a_id == "A3":
        use_pretrained = False
    elif a_id == "A4":
        use_crf = False
        
    # Load data
    train_data = load_conll(f"data/{task_type}_train.conll")
    val_data = load_conll(f"data/{task_type}_val.conll")
    test_data = load_conll(f"data/{task_type}_test.conll")
    
    with open("embeddings/word2idx.json", "r") as f:
        word2idx = json.load(f)
    if "<UNK>" not in word2idx: word2idx["<UNK>"] = 0
    
    all_tags = sorted(list(set([t for sent in train_data for t in sent[1]])))
    tag2idx = {t: i for i, t in enumerate(all_tags)}
    if use_crf:
        tag2idx["<START>"] = len(tag2idx)
        tag2idx["<STOP>"] = len(tag2idx)
        
    idx2tag = {i: t for t, i in tag2idx.items()}
    
    # Embeddings
    w2v_embeddings = None
    if use_pretrained:
        w2v_embeddings = np.load("embeddings/embeddings_w2v.npy")
        vocab_size = len(word2idx)
        embedding_dim = w2v_embeddings.shape[1]
    else:
        vocab_size = len(word2idx)
        embedding_dim = 100
        
    # Loaders
    train_ds = SequenceDataset(train_data, word2idx, tag2idx)
    val_ds = SequenceDataset(val_data, word2idx, tag2idx)
    test_ds = SequenceDataset(test_data, word2idx, tag2idx)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)
    
    # Model
    if use_crf:
        model = BiLSTM_CRF(vocab_size, len(tag2idx), embedding_dim, 128, 2, dropout, w2v_embeddings).to(device)
        if not bidirectional: # Manually adjust for A1 if needed, but BiLSTM_CRF uses BiLSTMModel
            model.bilstm.lstm = nn.LSTM(embedding_dim, 128, num_layers=2, dropout=dropout, bidirectional=False, batch_first=True).to(device)
            model.bilstm.hidden2tag = nn.Linear(128, len(tag2idx)).to(device)
        if use_pretrained:
            model.bilstm.embedding.weight.requires_grad = True
    else:
        model = BiLSTMModel(vocab_size, len(tag2idx), embedding_dim, 128, 2, dropout, w2v_embeddings).to(device)
        if not bidirectional:
            model.lstm = nn.LSTM(embedding_dim, 128, num_layers=2, dropout=dropout, bidirectional=False, batch_first=True).to(device)
            model.hidden2tag = nn.Linear(128, len(tag2idx)).to(device)
        if use_pretrained:
            model.embedding.weight.requires_grad = True
            
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=5, is_ner=use_crf)
    
    avg_loss, test_f1, targets, preds = evaluate_model(model, test_loader, criterion, is_ner=use_crf)
    
    print(f"Results for {a_id}: F1={test_f1:.4f}")
    return test_f1

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ablation_studies.py <A1|A2|A3|A4>")
        return
        
    a_id = sys.argv[1]
    res = run_ablation(a_id, "ner")
    print(f"FINAL_RESULT_{a_id}: {res}")

if __name__ == "__main__":
    main()
