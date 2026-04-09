import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from models.transformer import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY_NAMES = ["Politics", "Sports", "Economy", "International", "Health & Society"]

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
        ids = [self.word2idx.get(t, self.word2idx.get("<UNK>", 1)) for t in tokens]
        padding_len = self.max_len - len(ids)
        ids = ids + [0] * padding_len
        mask = [1] * (self.max_len - padding_len) + [0] * padding_len
        return (torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float), torch.tensor(item["label"], dtype=torch.long))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_class_weights(data_path, n_classes=5):
    with open(data_path) as f:
        data = json.load(f)
    counts = [0] * n_classes
    for item in data:
        counts[item['label']] += 1
    total = sum(counts)
    weights = [total / (n_classes * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float)

def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    if criterion is None:
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
    with open("embeddings/word2idx.json", "r") as f:
        word2idx = json.load(f)
    if "<UNK>" not in word2idx: word2idx["<UNK>"] = 1
    if "<PAD>" not in word2idx: word2idx["<PAD>"] = 0

    train_ds = ClassificationDataset("data/classification/train.json", word2idx)
    val_ds = ClassificationDataset("data/classification/val.json", word2idx)
    test_ds = ClassificationDataset("data/classification/test.json", word2idx)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    class_weights = get_class_weights("data/classification/train.json").to(device)
    model = TransformerClassifier(
        vocab_size=len(word2idx), d_model=128, n_heads=4, d_ff=512, n_layers=4, n_classes=5, dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    num_epochs = 100
    total_steps = len(train_loader) * num_epochs
    warmup_steps = 100
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, no_improve, patience = 0.0, 0, 20
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print(f"Training for {num_epochs} epochs with lr=5e-5 (Lower LR & Extended Epochs)")
    for epoch in range(num_epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)
        history["val_acc"].append(v_acc)
        print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val F1: {v_f1:.4f} | Val Acc: {v_acc:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), "models/transformer_cls.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("models/transformer_cls.pt", weights_only=True))
    test_loss, test_acc, test_f1, labels, preds = evaluate(model, test_loader)
    print(f"\nFinal Test Results | Acc: {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
    
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(CATEGORY_NAMES))
    ax.set_xticks(tick_marks); ax.set_xticklabels(CATEGORY_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(tick_marks); ax.set_yticklabels(CATEGORY_NAMES, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted Label', fontsize=12); ax.set_ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix — Transformer Classifier', fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/transformer_cm.png", dpi=150)
    plt.close()

    generate_attention_heatmaps(model, test_ds, word2idx)

def generate_attention_heatmaps(model, dataset, word2idx, n_articles=3, n_heads=2):
    model.eval()
    idx2word = {v: k for k, v in word2idx.items()}
    CATEGORY_NAMES = ["Politics", "Sports", "Economy", "International", "Health & Society"]
    correct_samples = []
    with torch.no_grad():
        for i in range(len(dataset)):
            ids, mask, label = dataset[i]
            ids_b = ids.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            logits, attentions = model(ids_b, mask_b)
            pred = torch.argmax(logits, dim=1).item()
            if pred == label.item():
                correct_samples.append((i, ids, mask, label.item(), attentions))
            if len(correct_samples) >= n_articles:
                break
    for sample_idx, (i, ids, mask, label, attentions) in enumerate(correct_samples):
        last_attn = attentions[-1].squeeze(0)  # (H, L+1, L+1)
        token_ids = ids.tolist()
        valid_len = sum(1 for t in token_ids if t != 0)
        display_len = min(valid_len + 1, 25)
        token_labels = ["[CLS]"] + [idx2word.get(tid, "<UNK>")[:8] for tid in token_ids[:display_len - 1]]
        fig, axes = plt.subplots(1, n_heads, figsize=(14, 6))
        if n_heads == 1: axes = [axes]
        for h in range(n_heads):
            attn_matrix = last_attn[h, :display_len, :display_len].cpu().numpy()
            ax = axes[h]
            im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=attn_matrix.max())
            ax.set_xticks(range(display_len)); ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
            ax.set_yticks(range(display_len)); ax.set_yticklabels(token_labels, fontsize=7)
            ax.set_xlabel("Key Tokens", fontsize=10); ax.set_ylabel("Query Tokens", fontsize=10)
            ax.set_title(f"Head {h+1}", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        true_cat = CATEGORY_NAMES[label]
        plt.suptitle(f"Article {sample_idx+1} — True: {true_cat} (Correct) — Final Layer", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"plots/attention_heatmap_article{sample_idx+1}.png", dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
