import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from models.sequence_labeler import BiLSTMModel, BiLSTM_CRF

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_conll(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        # Split by empty lines
        sents = f.read().strip().split("\n\n")
        for sent in sents:
            words = []
            tags = []
            for line in sent.split("\n"):
                parts = line.split()
                if len(parts) == 2:
                    words.append(parts[0])
                    tags.append(parts[1])
            if words:
                data.append((words, tags))
    return data

class SequenceDataset(Dataset):
    def __init__(self, data, word2idx, tag2idx):
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        words, tags = self.data[idx]
        word_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        tag_ids = [self.tag2idx[t] for t in tags]
        return torch.tensor(word_ids), torch.tensor(tag_ids), len(words)

def collate_fn(batch):
    word_ids, tag_ids, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    max_len = lengths.max()
    
    padded_words = torch.zeros(len(word_ids), max_len).long()
    padded_tags = torch.zeros(len(tag_ids), max_len).long()
    mask = torch.zeros(len(tag_ids), max_len).float()
    
    for i, (w, t, l) in enumerate(zip(word_ids, tag_ids, lengths)):
        padded_words[i, :l] = w
        padded_tags[i, :l] = t
        mask[i, :l] = 1.0
        
    return padded_words, padded_tags, lengths, mask

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=5, is_ner=False):
    best_f1 = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/temp_best.pt"
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            words, tags, lengths, mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            if is_ner:
                loss = model.neg_log_likelihood(words, tags, lengths, mask)
            else:
                tag_space = model(words, lengths)
                # Flatten for CE loss
                # tag_space: (B, L, T), tags: (B, L)
                loss = criterion(tag_space.view(-1, tag_space.size(-1)), tags.view(-1))
                # Mask padding
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_f1, _, _ = evaluate_model(model, val_loader, criterion, is_ner)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
                
    model.load_state_dict(torch.load(model_path))
    return history

def evaluate_model(model, data_loader, criterion, is_ner=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            words, tags, lengths, mask = [b.to(device) for b in batch]
            
            if is_ner:
                loss = model.neg_log_likelihood(words, tags, lengths, mask)
                preds = model.predict(words, lengths, mask)
                # preds is list of lists, tags is tensor
                for p, t, l in zip(preds, tags, lengths):
                    all_preds.extend(p)
                    all_targets.extend(t[:l].tolist())
            else:
                tag_space = model(words, lengths)
                loss = criterion(tag_space.view(-1, tag_space.size(-1)), tags.view(-1))
                loss = (loss * mask.view(-1)).sum() / mask.sum()
                
                preds = torch.argmax(tag_space, dim=2)
                for p, t, l in zip(preds, tags, lengths):
                    all_preds.extend(p[:l].tolist())
                    all_targets.extend(t[:l].tolist())
            
            total_loss += loss.item()
            
    avg_loss = total_loss / len(data_loader)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return avg_loss, macro_f1, all_targets, all_preds

def plot_curves(history, title, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{title} Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_f1"], label="Val F1")
    plt.title(f"{title} F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_task(task_type="pos", frozen=True):
    print(f"\n--- Running Task: {task_type.upper()} (Frozen Embeddings: {frozen}) ---")
    
    # Load data
    train_data = load_conll(f"data/{task_type}_train.conll")
    val_data = load_conll(f"data/{task_type}_val.conll")
    test_data = load_conll(f"data/{task_type}_test.conll")
    
    # Vocabulary and tags
    with open("embeddings/word2idx.json", "r") as f:
        word2idx = json.load(f)
    
    # Ensure <UNK> is in vocab (if not already)
    if "<UNK>" not in word2idx:
        word2idx["<UNK>"] = 0
        
    all_tags = sorted(list(set([t for sent in train_data for t in sent[1]])))
    if task_type == "ner":
        # Add START and STOP tags for CRF
        tag2idx = {t: i for i, t in enumerate(all_tags)}
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        tag2idx[START_TAG] = len(tag2idx)
        tag2idx[STOP_TAG] = len(tag2idx)
    else:
        tag2idx = {t: i for i, t in enumerate(all_tags)}
    
    idx2tag = {i: t for t, i in tag2idx.items()}
    
    # Pretrained embeddings
    w2v_embeddings = np.load("embeddings/embeddings_w2v.npy")
    # Wrap in nn.Embedding logic: check dimensions
    vocab_size = len(word2idx)
    embedding_dim = w2v_embeddings.shape[1]
    
    # Dataset and Loaders
    train_ds = SequenceDataset(train_data, word2idx, tag2idx)
    val_ds = SequenceDataset(val_data, word2idx, tag2idx)
    test_ds = SequenceDataset(test_data, word2idx, tag2idx)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)
    
    # Model
    if task_type == "ner":
        model = BiLSTM_CRF(vocab_size, len(tag2idx), embedding_dim, 128, 2, 0.5, w2v_embeddings).to(device)
        model.bilstm.embedding.weight.requires_grad = not frozen
    else:
        model = BiLSTMModel(vocab_size, len(tag2idx), embedding_dim, 128, 2, 0.5, w2v_embeddings).to(device)
        model.embedding.weight.requires_grad = not frozen
    
    criterion = nn.CrossEntropyLoss(reduction='none') # Masked later
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                          epochs=50, patience=5, is_ner=(task_type=="ner"))
    
    # Test Evaluation
    _, test_f1, targets, preds = evaluate_model(model, test_loader, criterion, is_ner=(task_type=="ner"))
    accuracy = accuracy_score(targets, preds)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(targets, preds, labels=range(len(all_tags)))
    
    # Save model
    suffix = "frozen" if frozen else "finetuned"
    torch.save(model.state_dict(), f"models/bilstm_{task_type}_{suffix}.pt")
    
    # Plot history
    plot_curves(history, f"{task_type.upper()} {suffix}", f"plots/{task_type}_{suffix}.png")
    
    return accuracy, test_f1, targets, preds, all_tags, history

def main():
    os.makedirs("plots", exist_ok=True)
    
    # 5.1 POS Tagging
    pos_frozen = run_task("pos", frozen=True)
    pos_tuned = run_task("pos", frozen=False)
    
    # Comparison table summary
    print("\n--- POS Comparison ---")
    print(f"Frozen: Acc={pos_frozen[0]:.4f}, F1={pos_frozen[1]:.4f}")
    print(f"Fine-tuned: Acc={pos_tuned[0]:.4f}, F1={pos_tuned[1]:.4f}")
    
    # 5.2 NER
    ner_frozen = run_task("ner", frozen=False) # Fine-tuned as default per requirement inference
    
    print("\n--- NER Results ---")
    print(f"Acc={ner_frozen[0]:.4f}, F1={ner_frozen[1]:.4f}")
    
    # Additional reporting logic for entities can be added here

if __name__ == "__main__":
    main()
