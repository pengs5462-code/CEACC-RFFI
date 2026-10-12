"""Audited OpenMax evaluation on frozen multiknown17 NetTime checkpoints."""
from __future__ import annotations
import argparse, importlib.util, json, sys, types
from pathlib import Path
import numpy as np
import torch
from scipy.stats import weibull_min
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-12

def load_nettime(root):
    spec = importlib.util.spec_from_file_location("model.layers", root / "layers.py")
    layers = importlib.util.module_from_spec(spec); spec.loader.exec_module(layers)
    package = types.ModuleType("model"); package.__path__ = []
    sys.modules["model"] = package; sys.modules["model.layers"] = layers
    spec = importlib.util.spec_from_file_location("archived_mran", root / "MRAN.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.NetTime

class NormalizedNetTime(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, value):
        if value.shape[-1] == 2: value = value.permute(0, 2, 1)
        value = value.float(); value = value - value.mean(2, keepdim=True)
        value = value / value.square().mean((1, 2), keepdim=True).add(1e-8).sqrt()
        return self.model(value)

def make_loader(x, y, index, mapping):
    labels = np.asarray([mapping.get(int(v), int(v)) for v in y[index]], dtype=np.int64)
    return DataLoader(TensorDataset(torch.from_numpy(np.asarray(x[index])), torch.from_numpy(labels)), batch_size=128)

@torch.no_grad()
def extract(model, loader, device):
    model.eval(); embeddings, logits, labels = [], [], []
    for value, target in loader:
        embedding, score = model(value.to(device)); embeddings.append(embedding.cpu().numpy()); logits.append(score.cpu().numpy()); labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(logits), np.concatenate(labels)

def eucos(value, mean):
    euclidean = np.linalg.norm(value - mean[None], axis=1) / 200.0
    denominator = np.linalg.norm(value, axis=1) * max(np.linalg.norm(mean), EPS)
    return euclidean + 1.0 - (value @ mean) / np.clip(denominator, EPS, None)

class OpenMax:
    def __init__(self, train_logits, train_labels, tail_size=20, alpha_rank=3):
        prediction = train_logits.argmax(1); self.means = []; self.weibull = []; self.correct_counts = []; self.tail_counts = []; self.alpha_rank = alpha_rank
        for class_id in range(train_logits.shape[1]):
            values = train_logits[(train_labels == class_id) & (prediction == class_id)]
            if len(values) < 3: raise ValueError(f"class {class_id}: fewer than three correct training samples")
            mean = values.mean(0); distance = np.sort(eucos(values, mean)); count = min(tail_size, len(distance))
            self.means.append(mean); self.weibull.append(weibull_min.fit(np.clip(distance[-count:], 1e-8, None), floc=0.0)); self.correct_counts.append(len(values)); self.tail_counts.append(count)
        self.means = np.asarray(self.means)
    def probabilities(self, logits):
        n, classes = logits.shape; distance = np.column_stack([eucos(logits, self.means[c]) for c in range(classes)])
        weibull = np.column_stack([weibull_min.cdf(distance[:, c], *self.weibull[c]) for c in range(classes)])
        rank = np.argsort(logits, axis=1)[:, ::-1]; alpha = np.zeros_like(logits, dtype=np.float64)
        for position in range(self.alpha_rank): alpha[np.arange(n), rank[:, position]] = (self.alpha_rank - position) / self.alpha_rank
        removed = logits * weibull * alpha; joined = np.concatenate((logits - removed, removed.sum(1, keepdims=True)), 1); joined -= joined.max(1, keepdims=True)
        probability = np.exp(joined); return probability / np.clip(probability.sum(1, keepdims=True), EPS, None)

def auc_rank(labels, scores):
    labels = labels.astype(int); order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores)); sorted_scores = scores[order]; start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]: end += 1
        ranks[order[start:end]] = (start + end + 1) / 2; start = end
    p = labels.sum(); n = len(labels) - p; return float((ranks[labels == 1].sum() - p * (p + 1) / 2) / (p * n))

def roc_points(known, rogue):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known, rogue]))[::-1], -np.inf]
    return np.asarray([np.mean(known >= t) for t in thresholds]), np.asarray([np.mean(rogue >= t) for t in thresholds])

def metric_row(validation, known, rogue, known_correct):
    threshold = float(np.quantile(validation, .9)); kr, rr = known > threshold, rogue > threshold
    tp, fp, fn = rr.sum(), kr.sum(), (~rr).sum(); precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    fpr, tpr = roc_points(known, rogue); index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    labels = np.r_[np.zeros(len(known), int), np.ones(len(rogue), int)]; scores = np.r_[known, rogue]; thresholds = np.r_[np.inf, np.sort(np.unique(scores)), -np.inf]
    ox = np.asarray([np.mean(rogue <= t) for t in thresholds]); oy = np.asarray([np.mean(known_correct & (known <= t)) for t in thresholds]); order = np.argsort(ox)
    return {"auc": auc_rank(labels, scores), "eer": float((fpr[index] + 1 - tpr[index]) / 2), "known_closed_acc": float(known_correct.mean()), "known_accept_rate": float((~kr).mean()), "unknown_recall": float(recall), "unknown_f1": float(2 * precision * recall / max(precision + recall, EPS)), "oscr": float(np.trapz(oy[order], ox[order])), "tau_known_only": threshold}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--dataset-root", required=True); p.add_argument("--split", type=int, required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--output-dir", required=True); p.add_argument("--device", default="cuda:0"); p.add_argument("--tail-size", type=int, default=20); p.add_argument("--alpha-rank", type=int, default=3); p.add_argument("--model-source-root", default=str(Path(__file__).resolve().parents[1] / "src" / "model")); a = p.parse_args()
    root = Path(a.dataset_root); manifest = json.loads((root / f"split_{a.split}.json").read_text()); known = list(map(int, manifest["known_classes"])); rogue = list(map(int, manifest["rogue_classes"])); mapping = {v: i for i, v in enumerate(known)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r"); y = np.load(root / "Y_global.npy"); role = np.load(root / "sample_role.npy"); km = np.isin(y, known)
    ti, vi, ki, ri = np.flatnonzero(km & (role == 0)), np.flatnonzero(km & (role == 1)), np.flatnonzero(km & (role == 2)), np.flatnonzero(np.isin(y, rogue) & (role == 2))
    checkpoint = torch.load(a.checkpoint, map_location="cpu"); NetTime = load_nettime(Path(a.model_source_root)); base = NetTime(c_in=2, c_out=len(known), nf=16); base.fexp = nn.Identity(); base.load_state_dict(checkpoint["model"])
    device = torch.device(a.device if torch.cuda.is_available() else "cpu"); model = NormalizedNetTime(base).to(device)
    train = extract(model, make_loader(x, y, ti, mapping), device); val = extract(model, make_loader(x, y, vi, mapping), device); kt = extract(model, make_loader(x, y, ki, mapping), device); rt = extract(model, make_loader(x, y, ri, {}), device)
    openmax = OpenMax(train[1], train[2], a.tail_size, a.alpha_rank); vp, kp, rp = openmax.probabilities(val[1]), openmax.probabilities(kt[1]), openmax.probabilities(rt[1]); vs, ks, rs = vp[:, -1], kp[:, -1], rp[:, -1]; pred = kp[:, :-1].argmax(1); result_metrics = metric_row(vs, ks, rs, pred == kt[2])
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); np.savez_compressed(out / "per_sample_scores.npz", validation_score=vs, known_score=ks, known_label=kt[2], known_prediction=pred, rogue_score=rs, rogue_label=rt[2])
    result = {"method": "OpenMax", "implementation": "audited_scipy_evt_on_frozen_nettime_v1", "split": a.split, "seed": a.seed, "known_classes": known, "rogue_classes": rogue, "metrics": result_metrics, "correct_train_counts": openmax.correct_counts, "weibull_tail_counts": openmax.tail_counts, "protocol": {"backbone": "exact frozen NetTime checkpoint used by Ours", "tail_size": a.tail_size, "alpha_rank": a.alpha_rank, "operating_threshold": "known-validation Q90", "target_rogue_used_for_fit_selection_or_threshold": False}}
    (out / "result.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)

if __name__ == "__main__": main()
