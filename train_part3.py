import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from models.transformer import TransformerClassifier

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationDataset(Dataset):
    def __init__(self, data_path, word2idx, max_len=256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["text"].split()[:self.max_len]
        ids = [self.word2idx.get(t, self.word2idx.get("<UNK>", 0)) for t in tokens]
        
        # Padding
        padding_len = self.max_len - len(ids)
        ids = ids + [0] * padding_len
        mask = [1] * (self.max_len - padding_len) + [0] * padding_len
        
        return torch.tensor(ids), torch.tensor(mask), torch.tensor(item["label"])

def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits, _ = model(ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1, all_labels, all_preds

def main():
    # Hydrate word2idx
    with open("embeddings/word2idx.json", "r") as f:
        word2idx = json.load(f)
    if "<UNK>" not in word2idx:
        word2idx["<UNK>"] = 0
        
    train_ds = ClassificationDataset("data/classification/train.json", word2idx)
    val_ds = ClassificationDataset("data/classification/val.json", word2idx)
    test_ds = ClassificationDataset("data/classification/test.json", word2idx)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)
    
    # Pretrained embeddings
    w2v_embeddings = np.load("embeddings/embeddings_w2v.npy")
    vocab_size = len(word2idx)
    d_model = w2v_embeddings.shape[1]
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        n_classes=5,
        dropout=0.2
    ).to(device)
    
    # Initialize with w2v
    model.embedding.weight.data.copy_(torch.tensor(w2v_embeddings))
    model.embedding.weight.requires_grad = True # Fine-tune
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    total_steps = len(train_loader) * 50 # 50 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    best_f1 = 0
    patience = 7
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    os.makedirs("models", exist_ok=True)
    print("Starting training...")
    
    for epoch in range(50):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader)
        
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)
        
        print(f"Epoch {epoch+1} | Train Loss: {t_loss:.4f} | Val F1: {v_f1:.4f} | Val Acc: {v_acc:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), "models/transformer_best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break
                
    # Final evaluation
    model.load_state_dict(torch.load("models/transformer_best.pt"))
    test_loss, test_acc, test_f1, labels, preds = evaluate(model, test_loader)
    print(f"\nFinal Test Results | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, range(5))
    plt.yticks(tick_marks, range(5))
    
    # Fill values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Transformer')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/transformer_cm.png")
    plt.close()
    
    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["val_f1"], label="Val F1")
    plt.legend()
    plt.savefig("plots/transformer_curves.png")
    plt.close()

if __name__ == "__main__":
    main()
