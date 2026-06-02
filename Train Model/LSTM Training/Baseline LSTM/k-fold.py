"""
k-fold.py  — 6-fold LOOCV for Baseline LSTM (23 visual features, no audio).

Leaves one game out per fold and reports per-game and per-class metrics.
Saves fold checkpoints as best_fold_N.pt in this directory.

Prerequisites:
  conda activate thesis_gpu
  python k-fold.py
"""
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

SCRIPT_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. CHECK CUDA / GPU STATUS
# ==========================================
print("\n" + "="*40)
print("SYSTEM CHECK")
print("="*40)
if torch.cuda.is_available():
    print(f"✅ CUDA available — GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("❌ CUDA not available — training on CPU (slow)")
    device = "cpu"
print("="*40 + "\n")

# ==========================================
# 2. CONFIGURATION
# ==========================================
DATA_ROOT = Path(r"C:\Users\USER\Desktop\Thesis\Paralkympics2024")
games = [
    "ISR_-_CAN_5-1",
    "TUR_-_BRA_3-1",
    "TUR_-_ISR_5-4",
    "ISR_-_BRA_8-4",
    "CHI_-_TUR_7-5",
    "BRA_-_TUR_3-3",
]

FEATURE_COLS = [
    "segment_type", "rel_t", "gap",
    "defender_seen", "thrower_seen", "ball_seen",
    "ball_x","ball_y","ball_dx","ball_dy","ball_w","ball_h","ball_conf",
    "thrower_x","thrower_y","thrower_w","thrower_h","thrower_conf",
    "defender_x","defender_y","defender_w","defender_h","defender_conf",
]
CLS = {('o',1):0, ('o',0):1, ('g',1):2, ('g',0):3, ('b',1):4, ('b',0):5}
TARGET_NAMES = ["o1", "o0", "g1", "g0", "b1", "b0"]
ALL_LABELS = [0, 1, 2, 3, 4, 5]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def calculate_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return specificity

def find_video_for_csv(csv_path: Path):
    game_dir = csv_path.parent.parent
    prefix = game_dir.name
    vids = list(game_dir.glob(f"{prefix}*.mp4")) + list(game_dir.glob(f"{prefix}*.mov"))
    return vids[0] if vids else None

def get_video_dims(video_path, fallback=(1920, 1080)):
    try:
        cap = cv2.VideoCapture(str(video_path))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w and h: return w, h
        return fallback
    except Exception:
        return fallback

def clean_throw(g: pd.DataFrame, W: float, H: float) -> pd.DataFrame:
    g.loc[g["label"] == 0, "outcome"] = g.loc[g["label"] == 0, "outcome"].fillna("o")
    g = g.sort_values("frame").reset_index(drop=True)
    g["ball_seen"] = (~g["ball_conf"].isna()).astype(int)
    g["thrower_seen"] = (~g["thrower_conf"].isna()).astype(int)
    g["defender_seen"] = (~g["defender_conf"].isna()).astype(int)
    cols_to_fill = ["ball_x","ball_y","ball_w","ball_h","ball_conf",
                    "thrower_x","thrower_y","thrower_w","thrower_h","thrower_conf",
                    "defender_x","defender_y","defender_w","defender_h","defender_conf"]
    existing = [c for c in cols_to_fill if c in g.columns]
    g[existing] = g[existing].ffill().bfill()
    for actor in ("ball", "thrower", "defender"):
        if f"{actor}_x" in g.columns:
            g[f"{actor}_x"] /= W;  g[f"{actor}_w"] /= W
            g[f"{actor}_y"] /= H;  g[f"{actor}_h"] /= H
    g["rel_t"] = g.index / (len(g) - 1) if len(g) > 1 else 0.0
    g["gap"] = g["frame"].diff().fillna(0)
    g[["ball_dx","ball_dy"]] = g[["ball_x","ball_y"]].diff().fillna(0)
    g["segment_type"] = (g["segment_type"] == "to").astype(int)
    return g

def preprocess_full(df_raw, W, H, uid_col="throw_uid"):
    df = df_raw.groupby(uid_col, group_keys=False).apply(clean_throw, W=W, H=H).reset_index(drop=True)
    def add_class(g):
        non_null = g["outcome"].dropna()
        outcome = non_null.iloc[0] if len(non_null) else "o"
        g["class_id"] = CLS.get((outcome, int(g["label"].iloc[0])), 0)
        return g
    df = df.groupby("throw_uid", group_keys=False).apply(add_class)
    return df.drop(columns=["frame","label","outcome"], errors='ignore')

def load_and_clean(csv_path: Path, dims_cache: dict) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    game_tag = csv_path.stem.split("_Throws")[0]
    raw["segment_uid"] = raw["segment_id"].astype(str).radd(f"{game_tag}_")
    raw["throw_uid"] = ((raw["segment_id"] + 1) // 2).astype(str).radd(f"{game_tag}_")
    vid = find_video_for_csv(csv_path)
    if csv_path not in dims_cache:
        dims_cache[csv_path] = get_video_dims(vid) if vid else (1920, 1080)
    W, H = dims_cache[csv_path]
    return preprocess_full(raw, W, H, uid_col="throw_uid")

def mirror_throw(df_one):
    df_flip = df_one.copy()
    for col in ["ball_x", "thrower_x", "defender_x"]:
        df_flip[col] = 1.0 - df_flip[col]
    df_flip["ball_dx"] *= -1
    return df_flip

# LSTM Model
class ThrowLSTM(nn.Module):
    def __init__(self, n_features=23, hidden=128, layers=2, n_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Sequential(nn.Linear(hidden*2, 64), nn.Tanh(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(hidden*2, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_classes))

    def forward(self, x, lengths):
        lengths_cpu = lengths.to('cpu')
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out_unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)
        B, L, _ = out_unpacked.size()
        idxs = torch.arange(L, device=out_unpacked.device).unsqueeze(0).expand(B, L)
        mask = idxs >= lengths.unsqueeze(1)
        scores = self.attn(out_unpacked).squeeze(-1).masked_fill(mask, float('-inf'))
        context = (out_unpacked * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.head(context)

class ThrowDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels):
        self.seqs = [torch.tensor(s) for s in seqs]
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx], self.labels[idx]

def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    return pad_sequence(xs, batch_first=True), lengths, torch.tensor(ys)

def make_sequences(df):
    seqs, labels = [], []
    for _, g in df.groupby("throw_uid"):
        seqs.append(g[FEATURE_COLS].values.astype("float32"))
        labels.append(int(g["class_id"].iloc[0]))
    return seqs, np.array(labels, dtype=np.int64)

# ==========================================
# 4. K-FOLD CROSS VALIDATION LOOP
# ==========================================
fold_results_list = []
global_preds = []
global_targets = []
dims_map = {}

print(f"Starting {len(games)}-Fold LOOCV  |  {len(FEATURE_COLS)} features per frame")

for fold_idx in range(len(games)):
    test_game_name = games[fold_idx]
    train_game_names = [g for i, g in enumerate(games) if i != fold_idx]

    print(f"\n{'='*40}")
    print(f"FOLD {fold_idx+1}/{len(games)} | Test Game: {test_game_name}")
    print(f"{'='*40}")

    test_csv = DATA_ROOT / test_game_name / "outputs" / f"{test_game_name}_Throws_lstm_training.csv"
    train_csvs = [DATA_ROOT / tg / "outputs" / f"{tg}_Throws_lstm_training.csv" for tg in train_game_names]
    train_csvs.extend([Path(f) for f in glob.glob(r"C:\Users\USER\Desktop\Thesis\Goals_Paralympics\outputs\*.csv")])

    print("-> Loading Data...")
    df_train = pd.concat([load_and_clean(p, dims_map) for p in train_csvs], ignore_index=True)
    df_test = load_and_clean(test_csv, dims_map)

    g1_id = CLS[('g',1)]
    flip_df = df_train.loc[df_train["class_id"] == g1_id].groupby("throw_uid", group_keys=False).apply(mirror_throw)
    if not flip_df.empty:
        flip_df["throw_uid"] = flip_df["throw_uid"].astype(str) + "_flip"
        df_train = pd.concat([df_train, flip_df], ignore_index=True)

    df_train[FEATURE_COLS] = df_train[FEATURE_COLS].fillna(0.0)
    df_test[FEATURE_COLS] = df_test[FEATURE_COLS].fillna(0.0)
    scaler = StandardScaler()
    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_test[FEATURE_COLS] = scaler.transform(df_test[FEATURE_COLS])

    train_seqs, train_y = make_sequences(df_train)
    test_seqs, test_y = make_sequences(df_test)

    freq = np.bincount(train_y, minlength=6)
    mask = freq > 0
    class_w = torch.zeros(6, device=device, dtype=torch.float32)
    class_w[mask] = torch.sqrt(torch.tensor(1.0 / freq[mask], device=device, dtype=torch.float32))
    class_w[2] *= 2.0

    sample_w = torch.tensor([1.0 / freq[l] for l in train_y], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, num_samples=len(train_y)*2, replacement=True)

    train_loader = torch.utils.data.DataLoader(ThrowDataset(train_seqs, train_y), batch_size=32, sampler=sampler, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(ThrowDataset(test_seqs, test_y), batch_size=32, shuffle=False, collate_fn=collate)

    model = ThrowLSTM(n_features=23).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    def focal_loss(logits, y, gamma=2):
        ce = nn.functional.cross_entropy(logits, y, reduction='none', weight=class_w)
        pt = torch.softmax(logits, 1).gather(1, y[:, None]).squeeze()
        return ((1 - pt) ** gamma * ce).mean()

    BEST_ACC = 0.0
    FOLD_CKPT = SCRIPT_DIR / f"best_fold_{fold_idx}.pt"
    patience = 10

    print("-> Training...")
    for epoch in range(30):
        model.train()
        for X, L, y in train_loader:
            X, L, y = X.to(device), L.to(device), y.to(device)
            optimizer.zero_grad()
            loss = focal_loss(model(X, L), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, L, y in test_loader:
                X, L, y = X.to(device), L.to(device), y.to(device)
                preds = model(X, L).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total > 0 else 0
        scheduler.step(val_acc)
        if val_acc > BEST_ACC:
            BEST_ACC = val_acc
            torch.save(model.state_dict(), FOLD_CKPT)
            patience = 10
        else:
            patience -= 1
            if patience == 0: break

    model.load_state_dict(torch.load(FOLD_CKPT))
    model.eval()
    fold_preds, fold_targets = [], []
    with torch.no_grad():
        for X, L, y in test_loader:
            X, L, y = X.to(device), L.to(device), y.to(device)
            preds = model(X, L).argmax(1)
            fold_preds.extend(preds.cpu().numpy())
            fold_targets.extend(y.cpu().numpy())

    global_preds.extend(fold_preds)
    global_targets.extend(fold_targets)

    accuracy = accuracy_score(fold_targets, fold_preds)
    _, _, f1_macro,  _ = precision_recall_fscore_support(fold_targets, fold_preds, average='macro',    labels=ALL_LABELS, zero_division=0)
    _, _, f1_weight, _ = precision_recall_fscore_support(fold_targets, fold_preds, average='weighted', labels=ALL_LABELS, zero_division=0)

    fold_results_list.append({
        "Game": test_game_name, "Accuracy": accuracy,
        "F1 (Macro)": f1_macro, "F1 (Weighted)": f1_weight,
    })
    print(f"   [Fold {fold_idx+1} Done] Acc: {accuracy:.4f}")

# ==========================================
# 5. METRICS REPORT
# ==========================================
print("\n" + "#"*60)
print("FINAL RESULTS  (Baseline LSTM — 23 visual features)")
print("#"*60 + "\n")

df_game_summary = pd.DataFrame(fold_results_list)
print("--- 1. METRICS PER GAME (TEST FOLD) ---")
print(df_game_summary.round(4))

precision, recall, f1, support = precision_recall_fscore_support(
    global_targets, global_preds, labels=ALL_LABELS, zero_division=0)
specificity = calculate_specificity(global_targets, global_preds, labels=ALL_LABELS)

df_class = pd.DataFrame({
    "Class": TARGET_NAMES, "Precision": precision, "Recall": recall,
    "Specificity": specificity, "F1-Score": f1, "Support": support,
})
print("\n--- 2. OVERALL METRICS BY CLASS ---")
print(df_class.round(4))

cm = confusion_matrix(global_targets, global_preds, labels=ALL_LABELS)
df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in TARGET_NAMES], columns=[f"Pred_{c}" for c in TARGET_NAMES])

# ==========================================
# 6. PLOTS
# ==========================================
print("\n--- Generating plots... ---")
plots_dir = SCRIPT_DIR / "loocv_plots_baseline"
plots_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.title('Confusion Matrix — Baseline LSTM (All Folds)')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(plots_dir / 'plot_cm.png'); plt.close()
print("  Saved: plot_cm.png")

df_melt = df_class.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1-Score", "Specificity"],
                        var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melt, x="Class", y="Score", hue="Metric", palette="viridis")
plt.title('Per-Class Metrics — Baseline LSTM'); plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7); plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(plots_dir / 'plot_per_class.png'); plt.close()
print("  Saved: plot_per_class.png")

df_game_melt = df_game_summary.melt(id_vars="Game", value_vars=["Accuracy", "F1 (Macro)", "F1 (Weighted)"],
                                    var_name="Metric", value_name="Score")
plt.figure(figsize=(14, 6))
sns.barplot(data=df_game_melt, x="Game", y="Score", hue="Metric", palette="magma")
plt.title('Per-Game Performance — Baseline LSTM'); plt.xticks(rotation=45)
plt.ylim(0, 1.1); plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / 'plot_per_game.png'); plt.close()
print("  Saved: plot_per_game.png")

print(f"\nAll plots saved to: {plots_dir}")
