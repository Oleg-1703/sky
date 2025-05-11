import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TS_FILE         = "data/time_series.xlsx"
LABELS_FILE     = "data/labels.csv"
MODEL_WEIGHTS   = "models/best_lstm_model.pth"
EMBEDDINGS_FILE = "data/ts_lstm_embeddings.csv"

EPOCHS            = 50
PATIENCE          = 7
BATCH_SIZE        = 32
HIDDEN_SIZE       = 128
LR                = 1e-4
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

# 1) Загрузка и подготовка данных
labels_df = pd.read_csv(LABELS_FILE)
labels_df = labels_df.drop_duplicates(subset="student_id").set_index("student_id")

xls = pd.ExcelFile(TS_FILE)
df  = xls.parse(xls.sheet_names[0])
df  = df[df["student_id"].isin(labels_df.index)]
df["ddl_date"] = pd.to_datetime(df["ddl_date"])
df = df.sort_values(["student_id", "ddl_date"])

seqs_grouped = df.groupby("student_id")["is_done"].apply(list)
student_ids_full = seqs_grouped.index.tolist()   # полный список ID
X = seqs_grouped.tolist()
y = labels_df.loc[student_ids_full, "is_refund"].values.astype(float)
assert len(X) == len(y), f"X и y должны быть одинаковой длины, {len(X)} != {len(y)}"

MAX_LEN = max(len(seq) for seq in X)
X_pad = np.array([seq + [0]*(MAX_LEN - len(seq)) for seq in X], dtype=float)

# 2) Делим на train/val, сразу разделяя и ID
student_ids_train, student_ids_val, \
X_train,        X_val, \
y_train,        y_val = train_test_split(
    student_ids_full, X_pad, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# 3) Dataset и DataLoader
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = SeqDataset(X_train, y_train)
val_ds   = SeqDataset(X_val,   y_val)

class_counts    = np.bincount(y_train.astype(int))
weights_per_cls = 1.0 / (class_counts + 1e-8)
sample_weights  = weights_per_cls[y_train.astype(int)]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# 4) Определяем модель, лосс, оптимизатор
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                # (batch, hidden_size)
        h = self.bn(h)
        h = self.dropout(h)
        return self.fc(h).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)
pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
criterion = FocalLoss(alpha=pos_weight, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# 5) Тренировка с ранней остановкой
epochs_no_improve = 0
best_f1, best_threshold = 0.0, 0.5

for epoch in range(1, EPOCHS+1):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(yb.numpy())

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    thresholds = np.linspace(0.1, 0.9, 17)
    f1s = [f1_score(all_targets, (probs>t).astype(int)) for t in thresholds]
    idx = np.argmax(f1s)
    val_f1, val_thr = f1s[idx], thresholds[idx]

    scheduler.step(val_f1)
    if val_f1 > best_f1:
        best_f1, best_threshold = val_f1, val_thr
        torch.save(model.state_dict(), MODEL_WEIGHTS)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch {epoch}/{EPOCHS} | Val F1: {val_f1:.4f} @thr={val_thr:.2f} | best: {best_f1:.4f} @thr={best_threshold:.2f}")

# 6) Извлечение эмбеддингов (без изменений)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()
full_ds    = SeqDataset(X_pad, y)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)

embs = []
with torch.no_grad():
    for Xb, _ in full_loader:
        Xb = Xb.to(device)
        _, (h_n, _) = model.lstm(Xb)
        embs.append(h_n[-1].cpu().numpy())

emb_np = np.vstack(embs)
emb_df = pd.DataFrame(emb_np, columns=[f'emb_{i}' for i in range(emb_np.shape[1])])
emb_df.insert(0, 'student_id', student_ids_full)
emb_df.to_csv(EMBEDDINGS_FILE, index=False)
print(f"Saved embeddings to {EMBEDDINGS_FILE}, best_val_f1={best_f1:.4f} @thr={best_threshold:.2f}")

# 7) Инференс по всему датасету и сохранение вероятностей
print("Loading best model and generating refund probabilities for all students...")
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()

refund_probs = []
with torch.no_grad():
    for Xb, _ in full_loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 0:
            probs = [probs]
        refund_probs.extend(probs)

df_probs = pd.DataFrame({
    'student_id': student_ids_full,
    'refund_prob': refund_probs
})
df_probs.to_csv('data/ts_lstm_refund_prob.csv', index=False)
print(f"Saved refund probabilities for {len(refund_probs)} students to ts_lstm_refund_prob.csv")
