#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCIFIT-Net: Scientific Fit Network for journal-venue recommendation
===================================================================
Input : Title + Abstract
Output: (1) Title-Abstract consistency, (2) Top-5 venue recommendation,
        (3) Scope compatibility / scope-mismatch risk of the best venue,
        (4) Explainability (key concepts, similar papers, DOIs)

Architecture (matches the proposed diagram)
-------------------------------------------
Stream A : Title encoder -> Abstract encoder -> Title<->Abstract cross-attention   (SciBERT, fine-tuned)
Stream B : SPECTER-2 scientific embedding of "title [SEP] abstract" -> projection   (frozen, cached)
Gated fusion (semantic + title-abstract alignment)  ->  paper representation h_p
Heads    : Venue recommender (multi-prototype journal profiles)
           Scope-mismatch estimator (paper-venue pair -> mismatch probability + scope distance)
           Consistency head (title-abstract consistency probability)

Usage
-----
  python scifit_net.py --mode all      --csv /path/Scopus_unique_data.csv   # prepare + train + evaluate + demo
  python scifit_net.py --mode eval                                          # re-evaluate saved model
  python scifit_net.py --mode predict --title "..." --abstract "..."        # single prediction

Kaggle: enable Internet (to download SciBERT / SPECTER-2 from the HF hub) and a GPU (T4).
"""
import os
import re
import sys
import json
import math
import time
import random
import argparse
import warnings
from dataclasses import dataclass, asdict
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, brier_score_loss,
                             precision_score, recall_score, roc_curve)
from scipy.stats import spearmanr, rankdata

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =====================================================================================
# 0. CONFIG
# =====================================================================================
@dataclass
class CFG:
    csv: str = ""                                    # auto-searched if empty / missing
    out_dir: str = "/kaggle/working/scifit_out" if os.path.isdir("/kaggle/working") else "./scifit_out"
    backbone: str = "allenai/scibert_scivocab_uncased"   # Stream A encoder
    specter: str = "allenai/specter2_base"               # Stream B encoder (frozen)

    # data
    min_papers: int = 10          # keep venues with >= this many papers (closed-set recommender)
    max_title: int = 48
    max_abs: int = 256
    max_spec: int = 320
    debug_n: int = 0              # >0: subsample papers (quick test)

    # model
    dim: int = 512
    n_proto: int = 3              # prototypes per venue (multi-prototype profile)
    heads: int = 8
    dropout: float = 0.1
    share_encoder: bool = True    # share SciBERT weights between title and abstract encoders
    grad_ckpt: bool = False

    # optimisation
    batch: int = 16
    epochs: int = 8
    lr_enc: float = 2e-5
    lr_head: float = 5e-4
    lr_proto: float = 5e-3
    wd: float = 0.01
    warmup: float = 0.06
    patience: int = 3
    label_smooth: float = 0.1
    fp16: bool = True
    workers: int = 2
    spec_batch: int = 64
    seed: int = 42

    # multi-task weights
    w_venue: float = 1.0
    w_cons: float = 0.5
    w_scope: float = 0.5
    w_cal: float = 1.0
    w_div: float = 0.05

    # self-supervised label construction
    p_corrupt: float = 0.3        # prob. a training paper is shown with a foreign abstract
    near_k: int = 10              # "near venue" list size
    scope_pos_thr: float = 0.5    # soft scope distance >= thr  -> "mismatch" for binary metrics

    # display thresholds
    cons_levels: tuple = (0.85, 0.65, 0.45)   # Highly / Consistent / Partially / Inconsistent
    scope_levels: tuple = (0.33, 0.66)        # mismatch prob: LOW / MEDIUM / HIGH risk


# =====================================================================================
# 1. UTILITIES
# =====================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def get_backbone(name, pretrained=True):
    try:
        if pretrained:
            return AutoModel.from_pretrained(name, add_pooling_layer=False)
        return AutoModel.from_config(AutoConfig.from_pretrained(name), add_pooling_layer=False)
    except TypeError:
        if pretrained:
            return AutoModel.from_pretrained(name)
        return AutoModel.from_config(AutoConfig.from_pretrained(name))


def find_csv(path):
    if path and os.path.isfile(path):
        return path
    roots = ["/kaggle/input", "/mnt/user-data/uploads", ".", os.path.expanduser("~")]
    for r in roots:
        for dp, _, fns in os.walk(r):
            if "Scopus_unique_data.csv" in fns:
                return os.path.join(dp, "Scopus_unique_data.csv")
    raise FileNotFoundError("Scopus_unique_data.csv not found. Pass --csv /path/to/file.csv")


def to_dev(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast_ctx(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=bool(enabled and device.type == "cuda"))


# =====================================================================================
# 2. DATA PREPARATION
# =====================================================================================
_COPY_RE = re.compile(r"\s*(?:Â)?©.*$", re.S)                      # "© 2024 Elsevier ... All rights reserved"
_COPY2_RE = re.compile(r"\s*Copyright\s*(?:\(c\)|©|\d{4}).*$", re.S | re.I)
_WS_RE = re.compile(r"\s+")


def read_csv_robust(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode CSV")


def clean_title(t):
    return _WS_RE.sub(" ", str(t)).strip()


def clean_abstract(a):
    """Removes publisher copyright tails (they leak the venue/publisher!) and extra whitespace."""
    a = str(a)
    a = _COPY_RE.sub("", a)
    a = _COPY2_RE.sub("", a)
    return _WS_RE.sub(" ", a).strip()


class Bundle:
    """Container for everything derived from the CSV."""
    pass


def load_and_prepare(cfg):
    path = find_csv(cfg.csv)
    print(f"[data] reading {path}")
    df = read_csv_robust(path)
    for c in ("Keywords", "DOI"):
        if c not in df.columns:
            df[c] = ""
    df = df[["Title", "Abstract", "Source title", "DOI", "Keywords"]].copy()
    df = df.dropna(subset=["Title", "Abstract", "Source title"])
    df["Title"] = df["Title"].map(clean_title)
    df["Abstract"] = df["Abstract"].map(clean_abstract)
    df = df[(df["Abstract"].str.split().str.len() >= 30) & (df["Title"].str.split().str.len() >= 3)]
    df = df.drop_duplicates("Abstract")
    df = df.assign(_tk=df["Title"].str.lower()).drop_duplicates("_tk").drop(columns="_tk")

    # canonical venue names (case / whitespace variants merged)
    df["venue_raw"] = df["Source title"].map(lambda s: _WS_RE.sub(" ", str(s)).strip())
    key = df["venue_raw"].str.lower()
    canon = df.groupby(key)["venue_raw"].agg(lambda s: s.value_counts().index[0])
    df["venue"] = key.map(canon)

    counts = df["venue"].value_counts()
    keep = counts[counts >= max(cfg.min_papers, 3)].index
    df = df[df["venue"].isin(keep)]
    if cfg.debug_n and len(df) > cfg.debug_n:
        df = df.sample(cfg.debug_n, random_state=cfg.seed)
        counts = df["venue"].value_counts()
        df = df[df["venue"].isin(counts[counts >= 3].index)]
    df = df.reset_index(drop=True)

    venues = sorted(df["venue"].unique().tolist())
    v2i = {v: i for i, v in enumerate(venues)}
    y = df["venue"].map(v2i).values.astype(np.int64)
    print(f"[data] papers={len(df)}  venues={len(venues)}  (min_papers={cfg.min_papers})")

    # stratified 80/10/10 split, per venue (guarantees >=1 paper per venue in val/test)
    rng = np.random.default_rng(cfg.seed)
    tr, va, te = [], [], []
    for c in range(len(venues)):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        n = len(ids)
        nv = max(1, int(round(0.1 * n)))
        tr += ids[2 * nv:].tolist()
        va += ids[:nv].tolist()
        te += ids[nv:2 * nv].tolist()
    B = Bundle()
    B.cfg = cfg
    B.df = df
    B.titles = df["Title"].tolist()
    B.abstracts = df["Abstract"].tolist()
    B.dois = df["DOI"].fillna("").astype(str).tolist()
    B.venues = venues
    B.y = y
    B.splits = {"train": np.array(sorted(tr)), "val": np.array(sorted(va)), "test": np.array(sorted(te))}
    B.train_counts = np.bincount(y[B.splits["train"]], minlength=len(venues))
    print({k: len(v) for k, v in B.splits.items()})
    return B


# ---------------------------- SPECTER-2 (Stream B) ------------------------------------
class SpecterEncoder:
    def __init__(self, name, device, max_len=320, fp16=True):
        self.tok = get_tokenizer(name)
        self.model = get_backbone(name, True).to(device).eval()
        self.device, self.max_len, self.fp16 = device, max_len, fp16

    @torch.no_grad()
    def encode(self, titles, abstracts, bs=64, desc="SPECTER-2"):
        sep = self.tok.sep_token or "[SEP]"
        texts = [f"{t}{sep}{a}" for t, a in zip(titles, abstracts)]
        order = np.argsort([len(x) for x in texts])
        out = None
        for s in tqdm(range(0, len(texts), bs), desc=desc):
            ids = order[s:s + bs]
            enc = self.tok([texts[i] for i in ids], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            with autocast_ctx(self.device, self.fp16):
                h = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
            h = h.float().cpu().numpy()
            if out is None:
                out = np.zeros((len(texts), h.shape[1]), dtype=np.float32)
            out[ids] = h
        return out


def make_negatives(spec_pair, splits, n_total, seed):
    """For every paper pick a foreign abstract from the SAME split.
       50% easy (random paper), 50% hard (one of the 10 nearest papers in SPECTER space)."""
    rng = np.random.default_rng(seed + 1)
    neg_idx = np.zeros(n_total, dtype=np.int64)
    neg_type = np.zeros(n_total, dtype=np.int64)   # 0 easy, 1 hard
    for _, ids in splits.items():
        n = len(ids)
        E = F.normalize(torch.tensor(spec_pair[ids]), dim=-1)
        k = min(10, n - 1)
        nn_list = []
        for s in range(0, n, 2048):
            sim = E[s:s + 2048] @ E.T
            r = torch.arange(sim.size(0))
            sim[r, r + s] = -2.0
            nn_list.append(sim.topk(k, dim=1).indices)
        nn_idx = torch.cat(nn_list).numpy()
        for local, i in enumerate(ids):
            if rng.random() < 0.5:
                j = local
                while j == local:
                    j = int(rng.integers(n))
                neg_type[i] = 0
            else:
                j = int(nn_idx[local, rng.integers(k)])
                neg_type[i] = 1
            neg_idx[i] = ids[j]
    return neg_idx, neg_type


def build_caches(B, cfg, device):
    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = f"min{cfg.min_papers}_seed{cfg.seed}_n{len(B.df)}"
    f = os.path.join(cfg.out_dir, f"cache_{tag}.npz")
    if os.path.isfile(f):
        z = np.load(f)
        print(f"[cache] loaded {f}")
        B.spec_pair, B.spec_neg, B.neg_idx, B.neg_type = z["spec_pair"], z["spec_neg"], z["neg_idx"], z["neg_type"]
        return
    enc = SpecterEncoder(cfg.specter, device, cfg.max_spec, cfg.fp16)
    B.spec_pair = enc.encode(B.titles, B.abstracts, cfg.spec_batch, "SPECTER-2 (true pairs)")
    B.neg_idx, B.neg_type = make_negatives(B.spec_pair, B.splits, len(B.df), cfg.seed)
    neg_abs = [B.abstracts[j] for j in B.neg_idx]
    B.spec_neg = enc.encode(B.titles, neg_abs, cfg.spec_batch, "SPECTER-2 (corrupted pairs)")
    np.savez_compressed(f, spec_pair=B.spec_pair, spec_neg=B.spec_neg, neg_idx=B.neg_idx, neg_type=B.neg_type)
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_venue_geometry(B, cfg):
    """Venue-venue scope distance from TRAIN-ONLY SPECTER centroids (used for self-supervised scope labels)."""
    C = len(B.venues)
    tr = B.splits["train"]
    cent = np.zeros((C, B.spec_pair.shape[1]), dtype=np.float64)
    for c in range(C):
        cent[c] = B.spec_pair[tr[B.y[tr] == c]].mean(0)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    Dm = 1.0 - cent @ cent.T
    off = ~np.eye(C, dtype=bool)
    Dn = np.zeros((C, C))
    Dn[off] = rankdata(Dm[off]) / off.sum()          # percentile-rank scope distance in (0,1]
    near = np.argsort(Dm + np.eye(C) * 9.0, axis=1)[:, :min(cfg.near_k, C - 1)]
    B.Dm, B.Dn, B.near, B.cent = Dm, Dn, near, cent


# =====================================================================================
# 3. DATASET / COLLATE
# =====================================================================================
class SciFitDataset(Dataset):
    """mode = 'train'  : random corruption (p_corrupt) + random scope-pair sampling
       mode = 'clean'  : true (title, abstract) pairs
       mode = 'corrupt': title + foreign abstract (fixed negatives)"""

    def __init__(self, B, split, mode, p_corrupt=0.3):
        self.B, self.ids, self.mode, self.p = B, B.splits[split], mode, p_corrupt
        self.C = len(B.venues)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        B = self.B
        i = int(self.ids[k])
        corrupt = (self.mode == "corrupt") or (self.mode == "train" and random.random() < self.p)
        if corrupt:
            j = int(B.neg_idx[i])
            abs_text, spec = B.abstracts[j], B.spec_neg[i]
        else:
            abs_text, spec = B.abstracts[i], B.spec_pair[i]
        c = int(B.y[i])
        c2, t, sv = c, 0.0, 0.0
        if not corrupt:
            sv = 1.0
            if self.mode == "train" and random.random() < 0.5:
                if random.random() < 0.5:
                    c2 = random.randrange(self.C)
                    while c2 == c:
                        c2 = random.randrange(self.C)
                else:
                    c2 = int(random.choice(B.near[c]))
                t = float(B.Dn[c, c2])
        return dict(title=B.titles[i], abstract=abs_text, spec=spec, y=c, corrupt=int(corrupt),
                    neg_type=int(B.neg_type[i]), c2=c2, t=t, sv=sv)


class Collator:
    def __init__(self, tok, cfg):
        self.tok, self.cfg = tok, cfg

    def __call__(self, items):
        tb = self.tok([x["title"] for x in items], max_length=self.cfg.max_title, truncation=True,
                      padding=True, return_tensors="pt")
        ab = self.tok([x["abstract"] for x in items], max_length=self.cfg.max_abs, truncation=True,
                      padding=True, return_tensors="pt")
        return dict(
            t_ids=tb["input_ids"], t_mask=tb["attention_mask"],
            a_ids=ab["input_ids"], a_mask=ab["attention_mask"],
            spec=torch.tensor(np.stack([x["spec"] for x in items]), dtype=torch.float32),
            y=torch.tensor([x["y"] for x in items], dtype=torch.long),
            corrupt=torch.tensor([x["corrupt"] for x in items], dtype=torch.float32),
            neg_type=torch.tensor([x["neg_type"] for x in items], dtype=torch.long),
            c2=torch.tensor([x["c2"] for x in items], dtype=torch.long),
            t=torch.tensor([x["t"] for x in items], dtype=torch.float32),
        )


# =====================================================================================
# 4. MODEL
# =====================================================================================
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, x, mask):
        w = self.q(x).squeeze(-1).float().masked_fill(mask == 0, -1e9)
        a = torch.softmax(w, dim=-1)
        return (a.unsqueeze(-1) * x.float()).sum(1), a


class CrossAttention(nn.Module):
    """Bidirectional title<->abstract cross-attention."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.t2a = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.a2t = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln_t, self.ln_a = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, T, Tm, A, Am):
        T, A = T.float(), A.float()
        t_ctx, _ = self.t2a(T, A, A, key_padding_mask=(Am == 0), need_weights=False)
        a_ctx, _ = self.a2t(A, T, T, key_padding_mask=(Tm == 0), need_weights=False)
        return self.ln_t(T + t_ctx), self.ln_a(A + a_ctx)


class SciFitNet(nn.Module):
    def __init__(self, cfg, n_venues, spec_dim, pretrained=True):
        super().__init__()
        self.cfg, self.C, self.K = cfg, n_venues, cfg.n_proto
        D = cfg.dim
        # ---- Stream A
        self.title_enc = get_backbone(cfg.backbone, pretrained)
        self.abs_enc = self.title_enc if cfg.share_encoder else get_backbone(cfg.backbone, pretrained)
        if cfg.grad_ckpt:
            self.title_enc.gradient_checkpointing_enable()
            if self.abs_enc is not self.title_enc:
                self.abs_enc.gradient_checkpointing_enable()
        H = self.title_enc.config.hidden_size
        self.xattn = CrossAttention(H, cfg.heads, cfg.dropout)
        self.pool_t, self.pool_a, self.pool_xt, self.pool_xa = AttnPool(H), AttnPool(H), AttnPool(H), AttnPool(H)
        self.stream_a = nn.Sequential(nn.Linear(4 * H, D), nn.GELU(), nn.LayerNorm(D), nn.Dropout(cfg.dropout))
        self.align = nn.Sequential(nn.Linear(2 * H + 2, D), nn.GELU(), nn.LayerNorm(D))
        # ---- Stream B
        self.stream_b = nn.Sequential(nn.LayerNorm(spec_dim), nn.Linear(spec_dim, D), nn.GELU(),
                                      nn.LayerNorm(D), nn.Dropout(cfg.dropout))
        # ---- Gated fusion
        self.gate = nn.Sequential(nn.Linear(3 * D, D), nn.Sigmoid())
        self.out_ln = nn.LayerNorm(D)
        # ---- Venue recommender: multi-prototype journal profiles
        self.protos = nn.Parameter(torch.randn(n_venues, cfg.n_proto, D) * 0.05)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.beta = 10.0
        self.cal_a = nn.Parameter(torch.tensor(8.0))      # compatibility calibration (sigmoid(a*s+b))
        self.cal_b = nn.Parameter(torch.tensor(-3.0))
        # ---- Scope-mismatch estimator
        self.scope_head = nn.Sequential(nn.Linear(4 * D + 1, 256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 1))
        # ---- Consistency head
        self.cons_head = nn.Sequential(nn.Linear(2 * D, 256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 1))

    # ------------------------------------------------------------------ encoder
    def forward(self, t_ids, t_mask, a_ids, a_mask, spec, need_attn=False):
        T = self.title_enc(input_ids=t_ids, attention_mask=t_mask).last_hidden_state
        A = self.abs_enc(input_ids=a_ids, attention_mask=a_mask).last_hidden_state
        XT, XA = self.xattn(T, t_mask, A, a_mask)
        t, _ = self.pool_t(T, t_mask)
        a, attn_a = self.pool_a(A, a_mask)
        xt, _ = self.pool_xt(XT, t_mask)
        xa, _ = self.pool_xa(XA, a_mask)
        sa = self.stream_a(torch.cat([t, a, xt, xa], -1))
        sb = self.stream_b(spec.float())
        c1 = F.cosine_similarity(t, a, dim=-1).unsqueeze(-1)
        c2 = F.cosine_similarity(xt, xa, dim=-1).unsqueeze(-1)
        al = self.align(torch.cat([(t - a).abs(), t * a, c1, c2], -1))
        g = self.gate(torch.cat([sa, sb, al], -1))
        h = self.out_ln(g * sa + (1 - g) * sb + al)
        return dict(h=h.float(), align=al.float(), attn_a=attn_a if need_attn else None)

    # ------------------------------------------------------------------ heads (call in fp32)
    def scale(self):
        return self.log_scale.exp().clamp(max=100.0)

    def venue_scores(self, h):
        """soft-max over prototypes of cosine similarity -> (B, C) venue score in ~[-1,1]"""
        hn = F.normalize(h, dim=-1)
        P = F.normalize(self.protos, dim=-1)
        cos = torch.einsum("bd,ckd->bck", hn, P)
        score = torch.logsumexp(self.beta * cos, dim=-1) / self.beta - math.log(self.K) / self.beta
        return score

    def compat(self, score):
        return torch.sigmoid(self.cal_a * score + self.cal_b)

    def cons_logit(self, h, align):
        return self.cons_head(torch.cat([h, align], -1)).squeeze(-1)

    def scope_logit(self, h, cidx):
        """paper h (B,D) x venue index (B,) -> (mismatch logit, scope distance)"""
        hn = F.normalize(h, dim=-1)
        P = F.normalize(self.protos[cidx].detach(), dim=-1)          # (B,K,D)
        v = F.normalize(P.mean(1), dim=-1)
        d = 1.0 - (P * hn.unsqueeze(1)).sum(-1).max(-1).values        # scope distance to closest prototype
        x = torch.cat([hn, v, (hn - v).abs(), hn * v, d.unsqueeze(-1)], -1)
        return self.scope_head(x).squeeze(-1), d

    def proto_div_loss(self):
        if self.K < 2:
            return self.protos.sum() * 0
        P = F.normalize(self.protos, dim=-1)
        G = torch.einsum("ckd,cjd->ckj", P, P)
        off = G * (1 - torch.eye(self.K, device=G.device))
        return off.clamp(min=0).sum() / (self.C * self.K * (self.K - 1))


# =====================================================================================
# 5. METRICS
# =====================================================================================
def ranking_metrics(scores, y, ks=(1, 3, 5, 10)):
    order = np.argsort(-scores, axis=1)
    ranks = (order == y[:, None]).argmax(1) + 1
    m = {f"Hit@{k}": float((ranks <= k).mean()) for k in ks}
    m["MRR"] = float((1.0 / ranks).mean())
    m["NDCG@5"] = float(np.where(ranks <= 5, 1.0 / np.log2(ranks + 1), 0.0).mean())
    m["MeanRank"] = float(ranks.mean())
    m["MedianRank"] = float(np.median(ranks))
    m["MacroF1@1"] = float(f1_score(y, order[:, 0], average="macro", zero_division=0))
    return m, ranks


def ece_score(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def eer_score(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2)


def best_f1_threshold(y, p):
    best, thr = -1, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best:
            best, thr = f, float(t)
    return thr


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================================
# 6. FEATURE EXTRACTION / HEAD EVALUATION HELPERS
# =====================================================================================
def make_loader(B, split, mode, cfg, tok, shuffle=False, bs=None):
    ds = SciFitDataset(B, split, mode, cfg.p_corrupt)
    return DataLoader(ds, batch_size=bs or cfg.batch * 2, shuffle=shuffle, num_workers=cfg.workers,
                      collate_fn=Collator(tok, cfg), drop_last=shuffle, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def extract(model, loader, device, fp16):
    model.eval()
    H, CL = [], []
    for b in loader:
        b = to_dev(b, device)
        with autocast_ctx(device, fp16):
            out = model(b["t_ids"], b["t_mask"], b["a_ids"], b["a_mask"], b["spec"])
        h, al = out["h"].float(), out["align"].float()
        CL.append(model.cons_logit(h, al).cpu())
        H.append(h.cpu())
    return torch.cat(H), torch.cat(CL).numpy()


@torch.no_grad()
def venue_scores_all(model, H, device, chunk=2048):
    model.eval()
    return torch.cat([model.venue_scores(H[s:s + chunk].to(device)).cpu() for s in range(0, len(H), chunk)]).numpy()


@torch.no_grad()
def scope_prob(model, H, venues, device, chunk=4096):
    model.eval()
    P, Dd = [], []
    venues = torch.as_tensor(venues)
    for s in range(0, len(H), chunk):
        lg, d = model.scope_logit(H[s:s + chunk].to(device), venues[s:s + chunk].to(device))
        P.append(torch.sigmoid(lg).cpu())
        Dd.append(d.cpu())
    return torch.cat(P).numpy(), torch.cat(Dd).numpy()


def build_scope_eval_pairs(y_split, B, seed):
    """4 pairs per paper: (true venue) + (near venue) + (random venue) + (far venue)  with soft targets."""
    rng = np.random.default_rng(seed + 7)
    C = len(B.venues)
    pos, ven, tgt, kind = [], [], [], []
    for k, c in enumerate(y_split):
        far = np.where(B.Dn[c] >= 0.8)[0]
        if len(far) == 0:
            far = np.argsort(-B.Dn[c])[:10]
        r = int(rng.integers(C))
        while r == c:
            r = int(rng.integers(C))
        for v, kd in ((c, 0), (int(rng.choice(B.near[c])), 1), (r, 2), (int(rng.choice(far)), 3)):
            pos.append(k)
            ven.append(v)
            tgt.append(0.0 if v == c else float(B.Dn[c, v]))
            kind.append(kd)
    return np.array(pos), np.array(ven), np.array(tgt), np.array(kind)


# =====================================================================================
# 7. EVALUATION
# =====================================================================================
def evaluate_split(model, B, cfg, tok, device, split, thr=None, verbose=True):
    """Full evaluation of the three heads. `thr` = thresholds tuned on validation (dict) or None (tune now)."""
    ids = B.splits[split]
    y = B.y[ids]
    H, cl_clean = extract(model, make_loader(B, split, "clean", cfg, tok), device, cfg.fp16)
    _, cl_corr = extract(model, make_loader(B, split, "corrupt", cfg, tok), device, cfg.fp16)
    res, new_thr = {}, {}

    # ---------------------------------------------------------- 1. venue recommendation
    S = venue_scores_all(model, H, device)
    vm, ranks = ranking_metrics(S, y)
    res["venue"] = vm
    # baselines for context
    cent_scores = B.spec_pair[ids] / (np.linalg.norm(B.spec_pair[ids], axis=1, keepdims=True) + 1e-9) @ B.cent.T
    res["venue_baseline_specter_centroid"] = ranking_metrics(cent_scores, y)[0]
    prior = np.tile(B.train_counts.astype(float), (len(ids), 1))
    res["venue_baseline_frequency_prior"] = ranking_metrics(prior + np.random.RandomState(0).rand(*prior.shape) * 1e-3, y)[0]
    # head / mid / tail venues
    fr = B.train_counts[y]
    bins = {"head(>=50 train papers)": fr >= 50, "mid(20-49)": (fr >= 20) & (fr < 50), "tail(<20)": fr < 20}
    res["venue_by_frequency"] = {}
    for name, m in bins.items():
        if m.sum() > 0:
            r = ranks[m]
            res["venue_by_frequency"][name] = dict(n=int(m.sum()), **{"Hit@1": float((r <= 1).mean()),
                                                                      "Hit@5": float((r <= 5).mean()),
                                                                      "MRR": float((1.0 / r).mean())})

    # ---------------------------------------------------------- 2. scope-mismatch estimator
    pos, ven, tgt, kind = build_scope_eval_pairs(y, B, cfg.seed)
    p_s, d_s = scope_prob(model, H[pos], ven, device)
    lab = (tgt >= cfg.scope_pos_thr).astype(int)      # 1 = mismatch
    sc = {}
    sc["AUROC"] = float(roc_auc_score(lab, p_s))
    sc["AUPRC"] = float(average_precision_score(lab, p_s))
    thr_s = (thr or {}).get("scope") or best_f1_threshold(lab, p_s)
    new_thr["scope"] = thr_s
    pred = (p_s >= thr_s).astype(int)
    sc.update({"threshold": thr_s, "F1": float(f1_score(lab, pred)), "Precision": float(precision_score(lab, pred, zero_division=0)),
               "Recall": float(recall_score(lab, pred)), "Accuracy": float((pred == lab).mean()),
               "Brier": float(brier_score_loss(lab, p_s)), "ECE": ece_score(p_s, lab),
               "MAE_vs_soft_target": float(np.abs(p_s - tgt).mean()),
               "Spearman_vs_soft_target": float(spearmanr(p_s, tgt)[0])})
    m_true, m_far, m_near = p_s[kind == 0], p_s[kind == 3], p_s[kind == 1]
    sc["Mean_mismatch_true_venue"] = float(m_true.mean())
    sc["Mean_mismatch_near_venue"] = float(m_near.mean())
    sc["Mean_mismatch_far_venue"] = float(m_far.mean())
    sc["AUROC_true_vs_far"] = float(roc_auc_score(np.r_[np.zeros(len(m_true)), np.ones(len(m_far))], np.r_[m_true, m_far]))
    sc["PairwiseOrder_true<far"] = float((m_true < m_far).mean())          # per paper (pairs are aligned)
    sc["PairwiseOrder_true<near"] = float((m_true < m_near).mean())
    # label-free check: does high mismatch for the recommended venue signal a wrong recommendation?
    top1 = S.argmax(1)
    p_top1, _ = scope_prob(model, H, top1, device)
    err = (top1 != y).astype(int)
    if 0 < err.sum() < len(err):
        sc["Top1_error_detection_AUROC"] = float(roc_auc_score(err, p_top1))
    sc["Mean_mismatch_top1_correct"] = float(p_top1[err == 0].mean()) if (err == 0).any() else float("nan")
    sc["Mean_mismatch_top1_wrong"] = float(p_top1[err == 1].mean()) if (err == 1).any() else float("nan")
    res["scope"] = sc

    # ---------------------------------------------------------- 3. title-abstract consistency
    p_clean, p_corr = sigmoid_np(cl_clean), sigmoid_np(cl_corr)
    ntype = B.neg_type[ids]
    yy = np.r_[np.ones(len(p_clean)), np.zeros(len(p_corr))].astype(int)
    pp = np.r_[p_clean, p_corr]
    co = {"AUROC": float(roc_auc_score(yy, pp)), "AUPRC": float(average_precision_score(yy, pp)),
          "EER": eer_score(yy, pp), "Brier": float(brier_score_loss(yy, pp)), "ECE": ece_score(pp, yy)}
    thr_c = (thr or {}).get("cons") or best_f1_threshold(yy, pp)
    new_thr["cons"] = thr_c
    pr = (pp >= thr_c).astype(int)
    co.update({"threshold": thr_c, "Accuracy@0.5": float(((pp >= 0.5).astype(int) == yy).mean()),
               "Accuracy@tuned": float((pr == yy).mean()), "F1@tuned": float(f1_score(yy, pr)),
               "Precision@tuned": float(precision_score(yy, pr, zero_division=0)),
               "Recall@tuned": float(recall_score(yy, pr))})
    for nm, mk in (("easy", ntype == 0), ("hard", ntype == 1)):
        if mk.any():
            co[f"AUROC_vs_{nm}_negatives"] = float(roc_auc_score(
                np.r_[np.ones(len(p_clean)), np.zeros(mk.sum())], np.r_[p_clean, p_corr[mk]]))
            co[f"Mean_score_{nm}_negatives"] = float(p_corr[mk].mean())
    co["Mean_score_true_pairs"] = float(p_clean.mean())
    hi = cfg.cons_levels[0]
    co["Frac_true_pairs_HighlyConsistent"] = float((p_clean >= hi).mean())
    co["Frac_corrupt_pairs_HighlyConsistent"] = float((p_corr >= hi).mean())
    res["consistency"] = co

    if verbose:
        print_results(res, split)
    return res, new_thr, H


def print_results(res, split):
    def block(title, d):
        print(f"\n  --- {title} ---")
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"    {k}: " + ", ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
            else:
                print(f"    {k:<38}{v:.4f}" if isinstance(v, float) else f"    {k:<38}{v}")
    print("\n" + "=" * 78 + f"\n  EVALUATION  [{split.upper()}]\n" + "=" * 78)
    block("1. VENUE RECOMMENDATION (SCIFIT-Net)", res["venue"])
    block("   baseline: SPECTER-2 centroid cosine", res["venue_baseline_specter_centroid"])
    block("   baseline: venue-frequency prior", res["venue_baseline_frequency_prior"])
    block("   SCIFIT-Net by venue frequency", res["venue_by_frequency"])
    block("2. SCOPE-MISMATCH ESTIMATOR", res["scope"])
    block("3. TITLE-ABSTRACT CONSISTENCY", res["consistency"])


# =====================================================================================
# 8. TRAINING
# =====================================================================================
def train(B, cfg, device):
    set_seed(cfg.seed)
    tok = get_tokenizer(cfg.backbone)
    model = SciFitNet(cfg, len(B.venues), B.spec_pair.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] SciFitNet params: {n_params:.1f}M")

    train_loader = make_loader(B, "train", "train", cfg, tok, shuffle=True, bs=cfg.batch)
    enc_params = list(model.title_enc.parameters()) + ([] if cfg.share_encoder else list(model.abs_enc.parameters()))
    enc_ids = {id(p) for p in enc_params}
    rest = [p for n, p in model.named_parameters() if id(p) not in enc_ids and n != "protos"]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg.lr_enc, "weight_decay": cfg.wd},
        {"params": rest, "lr": cfg.lr_head, "weight_decay": cfg.wd},
        {"params": [model.protos], "lr": cfg.lr_proto, "weight_decay": 0.0},
    ])
    total = cfg.epochs * len(train_loader)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup * total), total)
    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pos_w = torch.tensor(20.0, device=device)

    best, bad = -1.0, 0
    ckpt = os.path.join(cfg.out_dir, "best.pt")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0, agg, nb = time.time(), Counter(), 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{cfg.epochs}")
        for b in pbar:
            b = to_dev(b, device)
            with autocast_ctx(device, use_amp):
                out = model(b["t_ids"], b["t_mask"], b["a_ids"], b["a_mask"], b["spec"])
            h, al = out["h"].float(), out["align"].float()
            clean = b["corrupt"] == 0

            # consistency (all samples): label 1 = true pair, 0 = foreign abstract
            l_cons = F.binary_cross_entropy_with_logits(model.cons_logit(h, al), 1.0 - b["corrupt"])
            loss = cfg.w_cons * l_cons
            l_ven = l_scope = l_cal = torch.zeros((), device=device)
            if clean.any():
                hc, yc = h[clean], b["y"][clean]
                score = model.venue_scores(hc)
                l_ven = F.cross_entropy(model.scale() * score, yc, label_smoothing=cfg.label_smooth)
                onehot = F.one_hot(yc, len(B.venues)).float()
                l_cal = F.binary_cross_entropy_with_logits(model.cal_a * score.detach() + model.cal_b, onehot, pos_weight=pos_w)
                s_logit, _ = model.scope_logit(hc, b["c2"][clean])
                l_scope = F.binary_cross_entropy_with_logits(s_logit, b["t"][clean])
                loss = loss + cfg.w_venue * l_ven + cfg.w_scope * l_scope + cfg.w_cal * l_cal
            l_div = model.proto_div_loss()
            loss = loss + cfg.w_div * l_div

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            nb += 1
            for k, v in (("loss", loss), ("venue", l_ven), ("cons", l_cons), ("scope", l_scope)):
                agg[k] += float(v.detach())
            if nb % 20 == 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix({k: f"{agg[k] / nb:.3f}" for k in agg})

        # ----- validation
        res, _, _ = evaluate_split(model, B, cfg, tok, device, "val", verbose=False)
        v, c, s = res["venue"], res["consistency"], res["scope"]
        crit = v["MRR"] + 0.1 * c["AUROC"] + 0.1 * s["AUROC"]
        print(f"[ep {ep}] {time.time() - t0:.0f}s  train: " + " ".join(f"{k}={agg[k] / nb:.3f}" for k in agg) +
              f"\n         val: Hit@1={v['Hit@1']:.4f} Hit@5={v['Hit@5']:.4f} MRR={v['MRR']:.4f} NDCG@5={v['NDCG@5']:.4f} | "
              f"cons-AUC={c['AUROC']:.4f} | scope-AUC={s['AUROC']:.4f}")
        if crit > best:
            best, bad = crit, 0
            torch.save(model.state_dict(), ckpt)
            print("         * saved best")
        else:
            bad += 1
            if bad >= cfg.patience:
                print("[train] early stopping")
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, tok


# =====================================================================================
# 9. INFERENCE + EXPLAINABILITY + PRETTY OUTPUT
# =====================================================================================
_STOP = set("""a an the of in on at for to from by with without and or not is are was were be been being this that these those
it its as into than then which who whom whose their our we they he she can may might will would should could also
using use used based study studies results result paper method methods approach analysis however thus here show shown
found find new two one three between among during after before over under more most less per via within across""".split())


def key_concepts(tok, ids, attn, topn=6):
    toks = tok.convert_ids_to_tokens(list(ids))
    words, w = [], []
    for t, a in zip(toks, attn):
        if t in (tok.cls_token, tok.sep_token, tok.pad_token):
            continue
        if t.startswith("##") and words:
            words[-1][0] += t[2:]
            words[-1][1] = max(words[-1][1], float(a))
        else:
            words.append([t, float(a)])
    agg = {}
    for wd, a in words:
        if len(wd) < 4 or wd in _STOP or not re.match(r"^[a-z][a-z\-]+$", wd):
            continue
        agg[wd] = max(agg.get(wd, 0), a)
    return [w for w, _ in sorted(agg.items(), key=lambda x: -x[1])[:topn]]


class Predictor:
    def __init__(self, model, tok, cfg, B_meta, index, device, specter=None):
        self.model, self.tok, self.cfg, self.device = model.eval(), tok, cfg, device
        self.venues, self.index, self._specter = B_meta["venues"], index, specter

    @classmethod
    def load(cls, out_dir, device):
        meta = json.load(open(os.path.join(out_dir, "meta.json")))
        cfg = CFG(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["cfg"].items()})
        cfg.out_dir = out_dir
        model = SciFitNet(cfg, len(meta["venues"]), meta["spec_dim"], pretrained=False).to(device)
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
        z = np.load(os.path.join(out_dir, "index.npz"))
        im = json.load(open(os.path.join(out_dir, "index_meta.json")))
        index = dict(H=z["H"].astype(np.float32), y=z["y"], titles=im["titles"], dois=im["dois"])
        p = cls(model, get_tokenizer(cfg.backbone), cfg, meta, index, device)
        p.thr = meta.get("thresholds", {})
        return p

    def _spec(self):
        if self._specter is None:
            self._specter = SpecterEncoder(self.cfg.specter, self.device, self.cfg.max_spec, self.cfg.fp16)
        return self._specter

    @torch.no_grad()
    def predict(self, title, abstract, spec=None, topk=5):
        cfg, m = self.cfg, self.model
        title, abstract = clean_title(title), clean_abstract(abstract)
        if spec is None:
            spec = self._spec().encode([title], [abstract], bs=1, desc="SPECTER-2")[0]
        tb = self.tok([title], max_length=cfg.max_title, truncation=True, return_tensors="pt").to(self.device)
        ab = self.tok([abstract], max_length=cfg.max_abs, truncation=True, return_tensors="pt").to(self.device)
        sp = torch.tensor(spec[None], dtype=torch.float32, device=self.device)
        with autocast_ctx(self.device, cfg.fp16):
            out = m(tb["input_ids"], tb["attention_mask"], ab["input_ids"], ab["attention_mask"], sp, need_attn=True)
        h, al = out["h"].float(), out["align"].float()
        p_cons = float(torch.sigmoid(m.cons_logit(h, al))[0])
        S = m.venue_scores(h)[0]
        comp = m.compat(S)
        top = S.topk(min(topk, len(S))).indices
        best = int(top[0])
        lg, d = m.scope_logit(h, torch.tensor([best], device=self.device))
        mis = float(torch.sigmoid(lg)[0])
        concepts = key_concepts(self.tok, ab["input_ids"][0].cpu().numpy(), out["attn_a"][0].cpu().numpy())
        # similar papers (same venue first)
        hn = F.normalize(h, dim=-1).cpu().numpy()[0]
        Hn = self.index["H"]
        sims = Hn @ hn
        in_v = np.where(self.index["y"] == best)[0]
        cand = in_v if len(in_v) >= 3 else np.arange(len(sims))
        sel = cand[np.argsort(-sims[cand])[:3]]
        similar = [dict(title=self.index["titles"][i], doi=self.index["dois"][i], sim=float(sims[i])) for i in sel]
        return dict(consistency=p_cons, venues=[(self.venues[int(i)], float(comp[i])) for i in top],
                    best=self.venues[best], mismatch=mis, scope_distance=float(d[0]),
                    concepts=concepts, similar=similar)


def format_result(r, cfg, inner=50):
    hi, mid, lo = cfg.cons_levels
    c = r["consistency"]
    assess = "Highly Consistent" if c >= hi else "Consistent" if c >= mid else "Partially Consistent" if c >= lo else "Inconsistent"
    l1, l2 = cfg.scope_levels
    mis = r["mismatch"]
    risk = "LOW" if mis < l1 else "MEDIUM" if mis < l2 else "HIGH"
    compat = "HIGH" if mis < l1 else "MEDIUM" if mis < l2 else "LOW"

    def ln(t=""):
        return "║" + (" " + t).ljust(inner) + "║"
    top, mid_b, bot = "╔" + "═" * inner + "╗", "╠" + "═" * inner + "╣", "╚" + "═" * inner + "╝"
    L = [top, ln("SCIFIT-NET RESULT".center(inner - 2)), mid_b,
         ln("TITLE–ABSTRACT CONSISTENCY"), ln(), ln(f"Score: {c * 100:.1f}%"), ln(f"Assessment: {assess}"), mid_b,
         ln("VENUE RECOMMENDATION"), ln()]
    for i, (v, s) in enumerate(r["venues"], 1):
        v = v if len(v) <= 33 else v[:30] + "..."
        L.append(ln(f"{i}. {v:<33}{s * 100:>6.1f}%"))
    L += [mid_b, ln("BEST VENUE"), ln(), ln(r["best"][:inner - 2]), ln(),
          ln(f"Scope Compatibility: {compat}"), ln(f"Scope-Mismatch Risk: {risk}  (p={mis:.2f}, dist={r['scope_distance']:.2f})"), bot]
    txt = "\n".join(L)
    ex = ["", "EXPLAINABILITY", "  Key concepts : " + ", ".join(r["concepts"])]
    ex.append("  Similar papers in best venue:")
    for s in r["similar"]:
        ex.append(f"    - {s['title'][:80]}  [DOI: {s['doi'] or 'n/a'}]  (sim={s['sim']:.2f})")
    return txt + "\n" + "\n".join(ex)


@torch.no_grad()
def save_artifacts(model, B, cfg, tok, device, thresholds):
    H, _ = extract(model, make_loader(B, "train", "clean", cfg, tok), device, cfg.fp16)
    ids = B.splits["train"]
    np.savez_compressed(os.path.join(cfg.out_dir, "index.npz"), H=F.normalize(H, dim=-1).numpy().astype(np.float16), y=B.y[ids])
    json.dump(dict(titles=[B.titles[i] for i in ids], dois=[B.dois[i] for i in ids]),
              open(os.path.join(cfg.out_dir, "index_meta.json"), "w"))
    json.dump(dict(cfg=asdict(cfg), venues=B.venues, spec_dim=int(B.spec_pair.shape[1]), thresholds=thresholds),
              open(os.path.join(cfg.out_dir, "meta.json"), "w"))
    np.savez_compressed(os.path.join(cfg.out_dir, "venue_geometry.npz"), Dm=B.Dm, Dn=B.Dn, near=B.near)
    return dict(H=F.normalize(H, dim=-1).numpy(), y=B.y[ids], titles=[B.titles[i] for i in ids], dois=[B.dois[i] for i in ids])


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# =====================================================================================
# 10. MAIN
# =====================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "train", "eval", "predict"])
    for f in CFG.__dataclass_fields__.values():
        if f.type in (int, float, str):
            ap.add_argument(f"--{f.name}", type=f.type, default=f.default)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--no_share_encoder", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--title", default="")
    ap.add_argument("--abstract", default="")
    a, _ = ap.parse_known_args(argv)
    cfg = CFG(**{k: getattr(a, k) for k in CFG.__dataclass_fields__ if hasattr(a, k) and k not in
                 ("fp16", "share_encoder", "grad_ckpt", "cons_levels", "scope_levels")})
    cfg.fp16 = not a.no_fp16
    cfg.share_encoder = not a.no_share_encoder
    cfg.grad_ckpt = a.grad_ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"[env] device={device}  torch={torch.__version__}")
    set_seed(cfg.seed)

    if a.mode == "predict":
        P = Predictor.load(cfg.out_dir, device)
        print(format_result(P.predict(a.title, a.abstract), P.cfg))
        return

    B = load_and_prepare(cfg)
    build_caches(B, cfg, device)
    build_venue_geometry(B, cfg)

    if a.mode in ("all", "train"):
        model, tok = train(B, cfg, device)
    else:
        P = Predictor.load(cfg.out_dir, device)
        model, tok = P.model, P.tok

    # ---- final evaluation (thresholds tuned on VAL, applied to TEST)
    _, thr, _ = evaluate_split(model, B, cfg, tok, device, "val", verbose=False)
    res, _, _ = evaluate_split(model, B, cfg, tok, device, "test", thr=thr, verbose=True)
    json.dump(to_py(res), open(os.path.join(cfg.out_dir, "test_results.json"), "w"), indent=2)
    pd.DataFrame([{"Task": "Venue", **{k: v for k, v in res["venue"].items()}}]).to_csv(
        os.path.join(cfg.out_dir, "test_venue_metrics.csv"), index=False)

    index = save_artifacts(model, B, cfg, tok, device, thr)
    # ---- demo on 3 random test papers
    P = Predictor(model, tok, cfg, dict(venues=B.venues), index, device)
    rng = np.random.default_rng(cfg.seed)
    for i in rng.choice(B.splits["test"], 3, replace=False):
        print("\n" + "-" * 78 + f"\nTITLE: {B.titles[i]}\nTRUE VENUE: {B.venues[B.y[i]]}")
        print(format_result(P.predict(B.titles[i], B.abstracts[i], spec=B.spec_pair[i]), cfg))
    print(f"\n[done] artifacts in {cfg.out_dir}")


if __name__ == "__main__":
    main()