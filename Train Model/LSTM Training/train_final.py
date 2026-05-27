"""
train_final.py - Final deployment LSTM model trained on ALL 6 games + goal clips.

Saves the best checkpoint (by accuracy) to SAVE_PATH.

Run
---
  python "Train Model/LSTM Training/train_final.py"

Data layout expected
--------------------
  DATA_ROOT/<GAME>/outputs/<GAME>_Throws_lstm_training.csv   (per-game LSTM CSVs)
  GOALS_DIR/*.csv                                            (goal-clip CSVs)

These CSVs are produced by predict_pipeline_with_YAMNet.py (or yolo_cnn_predict_2.py).
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import cv2

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent    # Train Model/LSTM Training/
_REPO = _HERE.parent.parent               # GoalBall/

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\path\to\Paralkympics2024")       # folder with per-game sub-dirs
GOALS_DIR = Path(r"C:\path\to\Goals_Paralympics\outputs")  # goal-clip CSV folder
SAVE_PATH = _REPO / "Model Weights" / "final_model.pt" # where to save the trained model
# ──────────────────────────────────────────────────────────────────────────────

GAMES = ["ISR_-_CAN_5-1","TUR_-_BRA_3-1","TUR_-_ISR_5-4","ISR_-_BRA_8-4","CHI_-_TUR_7-5","BRA_-_TUR_3-3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS=100; PATIENCE=10; BATCH=32; LR=1e-3; WD=1e-4; GAMMA=2

BALL_COLS=["ball_x","ball_y","ball_w","ball_h","ball_conf"]
THROWER_COLS=["thrower_x","thrower_y","thrower_w","thrower_h","thrower_conf"]
DEFENDER_COLS=["defender_x","defender_y","defender_w","defender_h","defender_conf"]
FEATURE_COLS=["segment_type","rel_t","gap","defender_seen","thrower_seen","ball_seen",
    "ball_x","ball_y","ball_dx","ball_dy","ball_w","ball_h","ball_conf",
    "thrower_x","thrower_y","thrower_w","thrower_h","thrower_conf",
    "defender_x","defender_y","defender_w","defender_h","defender_conf"]
CLS={('o',1):0,('o',0):1,('g',1):2,('g',0):3,('b',1):4,('b',0):5}

print(f"Device: {DEVICE}")

def find_video(p):
    gd=p.parent.parent; v=list(gd.glob(f"{gd.name}*.mp4"))+list(gd.glob(f"{gd.name}*.mov"))
    return v[0] if v else None
def get_dims(vp,fb=(1920,1080)):
    if vp is None: return fb
    cap=cv2.VideoCapture(str(vp)); w=int(cap.get(3)); h=int(cap.get(4)); cap.release()
    return (w,h) if w and h else fb

DIMS={}
for g in GAMES:
    csv=DATA_ROOT/g/"outputs"/f"{g}_Throws_lstm_training.csv"; DIMS[csv]=get_dims(find_video(csv))
for f in sorted(GOALS_DIR.glob("*.csv")): DIMS[f]=get_dims(find_video(f))

def clean_throw(g,W,H):
    g=g.copy(); g.loc[g["label"]==0,"outcome"]=g.loc[g["label"]==0,"outcome"].fillna("o")
    g=g.sort_values("frame").reset_index(drop=True)
    g["ball_seen"]=(~g["ball_conf"].isna()).astype(int)
    g["thrower_seen"]=(~g["thrower_conf"].isna()).astype(int)
    g["defender_seen"]=(~g["defender_conf"].isna()).astype(int)
    for cols in (BALL_COLS,THROWER_COLS,DEFENDER_COLS): g[cols]=g[cols].ffill().bfill()
    for a in ("ball","thrower","defender"):
        g[f"{a}_x"]/=W; g[f"{a}_w"]/=W; g[f"{a}_y"]/=H; g[f"{a}_h"]/=H
    g["rel_t"]=g.index/(len(g)-1) if len(g)>1 else 0.0
    g["gap"]=g["frame"].diff().fillna(0)
    g[["ball_dx","ball_dy"]]=g[["ball_x","ball_y"]].diff().fillna(0)
    g["segment_type"]=(g["segment_type"]=="to").astype(int); return g

def preprocess(df_raw,W,H):
    df=(df_raw.groupby("throw_uid",group_keys=False).apply(clean_throw,W=W,H=H).reset_index(drop=True))
    def ac(g):
        nn2=g["outcome"].dropna(); out=nn2.iloc[0] if len(nn2) else "o"
        g["class_id"]=CLS[(out,int(g["label"].iloc[0]))]; return g
    df=df.groupby("throw_uid",group_keys=False).apply(ac)
    return df.drop(columns=["frame","label","outcome"])

def load_csv(csv_path):
    raw=pd.read_csv(csv_path); tag=csv_path.stem.split("_Throws")[0]
    raw["segment_uid"]=raw["segment_id"].astype(str).radd(f"{tag}_")
    raw["throw_uid"]=((raw["segment_id"]+1)//2).astype(str).radd(f"{tag}_")
    return preprocess(raw,DIMS[csv_path][0] if isinstance(DIMS[csv_path],tuple) else DIMS[csv_path][0],
                      DIMS[csv_path][1])

def load_csv2(csv_path):
    raw=pd.read_csv(csv_path); tag=csv_path.stem.split("_Throws")[0]
    raw["segment_uid"]=raw["segment_id"].astype(str).radd(f"{tag}_")
    raw["throw_uid"]=((raw["segment_id"]+1)//2).astype(str).radd(f"{tag}_")
    W,H=DIMS[csv_path]; return preprocess(raw,W,H)

def mirror_g1(df):
    def _m(d):
        d=d.copy()
        for c in ["ball_x","thrower_x","defender_x"]: d[c]=1.0-d[c]
        d["ball_dx"]*=-1; return d
    gid=CLS[('g',1)]; flip=(df[df["class_id"]==gid].groupby("throw_uid",group_keys=False).apply(_m))
    flip["throw_uid"]=flip["throw_uid"].astype(str)+"_flip"
    return pd.concat([df,flip],ignore_index=True)

class ThrowDataset(Dataset):
    def __init__(self,s,l): self.seqs=[torch.tensor(x) for x in s]; self.labels=torch.tensor(l)
    def __len__(self): return len(self.seqs)
    def __getitem__(self,i): return self.seqs[i],self.labels[i]

def collate(batch):
    xs,ys=zip(*batch); lengths=torch.tensor([len(x) for x in xs])
    return pad_sequence(xs,batch_first=True),lengths,torch.tensor(ys)

class ThrowLSTM(nn.Module):
    def __init__(self,n=23,h=128,l=2,c=6):
        super().__init__()
        self.lstm=nn.LSTM(n,h,l,batch_first=True,bidirectional=True,dropout=0.3)
        self.attn=nn.Sequential(nn.Linear(h*2,64),nn.Tanh(),nn.Dropout(0.2),nn.Linear(64,1))
        self.head=nn.Sequential(nn.Linear(h*2,128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,c))
    def forward(self,x,lengths):
        lc=lengths.to(dtype=torch.int64,device="cpu")
        packed=pack_padded_sequence(x,lc,batch_first=True,enforce_sorted=False)
        out,_=self.lstm(packed); out,_=pad_packed_sequence(out,batch_first=True)
        B,L,_=out.size(); idxs=torch.arange(L,device=out.device).unsqueeze(0).expand(B,L)
        mask=idxs>=lc.to(out.device).unsqueeze(1)
        scores=self.attn(out).squeeze(-1).masked_fill(mask,float("-inf"))
        context=(out*torch.softmax(scores,dim=1).unsqueeze(-1)).sum(dim=1)
        return self.head(context)

print("Loading all data...")
parts=[]
for g in GAMES:
    csv=DATA_ROOT/g/"outputs"/f"{g}_Throws_lstm_training.csv"
    df=load_csv2(csv); print(f"  {g}: {df['throw_uid'].nunique()} throws"); parts.append(df)
for f in sorted(GOALS_DIR.glob("*.csv")): parts.append(load_csv2(f))
df_all=mirror_g1(pd.concat(parts,ignore_index=True))
print(f"Total after augmentation: {df_all['throw_uid'].nunique()} throws")
print(df_all.groupby("throw_uid")["class_id"].first().value_counts().sort_index())

df_all[FEATURE_COLS]=df_all[FEATURE_COLS].fillna(0)
sc=StandardScaler(); df_all[FEATURE_COLS]=sc.fit_transform(df_all[FEATURE_COLS])
seqs,labels=[],[]
for _,g in df_all.groupby("throw_uid"):
    seqs.append(g[FEATURE_COLS].values.astype("float32")); labels.append(int(g["class_id"].iloc[0]))
labels=np.array(labels,dtype=np.int64)
print(f"Total sequences: {len(seqs)}")

freq=np.bincount(labels,minlength=6); mask=freq>0
cw=torch.zeros(6,dtype=torch.float32,device=DEVICE)
cw[mask]=torch.sqrt(torch.tensor(1.0/freq[mask],dtype=torch.float32,device=DEVICE)); cw[2]*=2
def focal(logits,y):
    ce=nn.functional.cross_entropy(logits,y,reduction="none",weight=cw)
    pt=torch.softmax(logits,1).gather(1,y[:,None]).squeeze()
    return ((1-pt)**GAMMA*ce).mean()

sw=torch.tensor([1.0/freq[l] for l in labels],dtype=torch.float32)
sampler=WeightedRandomSampler(sw,len(labels)*3,replacement=True)
loader=DataLoader(ThrowDataset(seqs,labels),batch_size=BATCH,sampler=sampler,collate_fn=collate)

model=ThrowLSTM().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.3,patience=3)
prec_m=MulticlassPrecision(num_classes=6,average="none").to(DEVICE)
rec_m=MulticlassRecall(num_classes=6,average="none").to(DEVICE)
f1_m=MulticlassF1Score(num_classes=6,average="macro").to(DEVICE)

BEST=0.0; cooldown=PATIENCE
print(f"\nTraining on ALL data...\n{'ep':>4}  {'lr':>8}  {'acc':>6}  {'f1':>6}  {'prec_g1':>8}  {'rec_g1':>7}")
for ep in range(1,N_EPOCHS+1):
    model.train(); tot=correct=loss_sum=0; prec_m.reset(); rec_m.reset(); f1_m.reset()
    for X,lengths,y in loader:
        X,lengths,y=X.to(DEVICE),lengths.to(DEVICE),y.to(DEVICE)
        opt.zero_grad(); logits=model(X,lengths); loss=focal(logits,y)
        loss.backward(); opt.step()
        loss_sum+=loss.item()*y.size(0); pred=logits.argmax(1)
        correct+=(pred==y).sum().item(); tot+=y.size(0)
        prec_m.update(pred,y); rec_m.update(pred,y); f1_m.update(pred,y)
    acc=correct/tot; pg1=prec_m.compute()[2].item(); rg1=rec_m.compute()[2].item(); f1=f1_m.compute().item()
    sched.step(acc); lr_now=opt.param_groups[0]["lr"]
    print(f"{ep:>4}  {lr_now:>8.1e}  {acc:>6.3f}  {f1:>6.3f}  {pg1:>8.3f}  {rg1:>7.3f}")
    if acc>BEST+1e-9:
        BEST=acc; torch.save(model.state_dict(),SAVE_PATH)
        print(f"       [SAVED] acc={BEST:.4f}"); cooldown=PATIENCE
    else:
        cooldown-=1
        if cooldown==0: print(f"[STOP] ep{ep} best={BEST:.4f}"); break

print(f"\nFinal model → {SAVE_PATH}\nBest acc: {BEST:.4f}")
