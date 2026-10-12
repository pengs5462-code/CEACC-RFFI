"""NS-RFF author-architecture adaptation for multiknown17.

Uses the public CNN-L2Softmax model and optimization from xrj-com/NS-RFF.
The authors' network is defined for a 1,280-complex-sample ZigBee preamble, so
the deterministic leading 1,280 samples are used for every 4,800-sample record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_hyperrsi_official import dataset_fingerprint, metrics


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class ArcMarginProduct(nn.Module):
    """Official NS-RFF L2Softmax head (s=10, m=0)."""
    def __init__(self, in_features, out_features, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, value, target=None):
        cosine = F.linear(F.normalize(value), F.normalize(self.weight))
        return cosine * self.scale


class NSRFFCNN(nn.Module):
    """CLF_L2Softmax/BaseCLF2 from the authors' repository."""
    def __init__(self, classes, width=24, embedding_dim=512):
        super().__init__()
        self.features_net = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1), nn.BatchNorm2d(width), nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.BatchNorm2d(width * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 4, width * 8, 3, padding=1), nn.BatchNorm2d(width * 8), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 8, width * 16, 3, stride=2, padding=1), nn.BatchNorm2d(width * 16), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 16, width * 32, 3, stride=2, padding=1), nn.BatchNorm2d(width * 32), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 32, embedding_dim, (2, 10)),
        )
        self.output = ArcMarginProduct(embedding_dim, classes, 10.0)

    def features(self, value):
        # Official reshape: two I/Q channels laid out on a 16 x 80 grid.
        image = value.reshape(len(value), 2, 16, 80)
        return self.features_net(image).reshape(len(value), -1)

    def forward(self, value, target=None):
        embedding = self.features(value)
        return embedding, self.output(embedding, target)


def make_loader(x, y, indices, mapping, batch_size, shuffle):
    values = np.asarray(x[indices, :1280, :], dtype=np.float32)
    values = np.transpose(values, (0, 2, 1))
    values -= values.mean(axis=2, keepdims=True)
    values /= np.sqrt(np.mean(values * values, axis=(1, 2), keepdims=True) + 1e-8)
    labels = np.asarray([mapping.get(int(v), int(v)) for v in y[indices]], dtype=np.int64)
    return DataLoader(TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
                      batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def extract(model, loader, device):
    model.eval(); embeddings, labels = [], []
    for value, target in loader:
        embedding, _ = model(value.to(device))
        embeddings.append(F.normalize(embedding, dim=1).cpu().numpy()); labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def score(embedding, prototypes):
    similarity = embedding @ prototypes.T
    return -similarity.max(1), similarity.argmax(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True); parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=24); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    args = parser.parse_args(); seed_everything(args.seed)
    root = Path(args.dataset_root); manifest = json.loads((root / f"split_{args.split}.json").read_text())
    known = list(map(int, manifest["known_classes"])); rogue = list(map(int, manifest["rogue_classes"]))
    mapping = {value: index for index, value in enumerate(known)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r"); y = np.load(root / "Y_global.npy"); role = np.load(root / "sample_role.npy")
    known_mask = np.isin(y, known)
    index = {"train": np.flatnonzero(known_mask & (role == 0)), "validation": np.flatnonzero(known_mask & (role == 1)),
             "known": np.flatnonzero(known_mask & (role == 2)), "rogue": np.flatnonzero(np.isin(y, rogue) & (role == 2))}
    train = make_loader(x, y, index["train"], mapping, args.batch_size, True)
    validation = make_loader(x, y, index["validation"], mapping, 128, False)
    known_test = make_loader(x, y, index["known"], mapping, 128, False)
    rogue_test = make_loader(x, y, index["rogue"], {}, 128, False)
    fingerprints = {name: dataset_fingerprint(loader) for name, loader in (("known_train", train), ("known_validation", validation), ("known_test", known_test))}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = NSRFFCNN(len(known)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.99))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); best_path = output / "best.pt"
    best, stale, history, start = float("inf"), 0, [], time.time()
    for epoch in range(args.epochs):
        model.train(); train_loss = train_correct = train_count = 0
        for value, target in train:
            value, target = value.to(device), target.to(device); optimizer.zero_grad(set_to_none=True)
            _, logits = model(value, target); loss = F.cross_entropy(logits, target); loss.backward(); optimizer.step()
            train_loss += loss.item() * len(target); train_correct += (logits.argmax(1) == target).sum().item(); train_count += len(target)
        model.eval(); val_loss = val_correct = val_count = 0
        with torch.no_grad():
            for value, target in validation:
                value, target = value.to(device), target.to(device); _, logits = model(value)
                loss = F.cross_entropy(logits, target); val_loss += loss.item() * len(target)
                val_correct += (logits.argmax(1) == target).sum().item(); val_count += len(target)
        val_loss /= val_count
        row = {"epoch": epoch + 1, "train_loss": train_loss / train_count, "train_acc": train_correct / train_count,
               "val_loss": val_loss, "val_acc": val_correct / val_count}; history.append(row); print(json.dumps(row), flush=True)
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0; torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss}, best_path)
        else:
            stale += 1
            if stale >= args.patience: break
    saved = torch.load(best_path, map_location=device); model.load_state_dict(saved["model"])
    train_embedding, train_label = extract(model, DataLoader(train.dataset, batch_size=128), device)
    val_embedding, _ = extract(model, validation, device); known_embedding, known_label = extract(model, known_test, device)
    rogue_embedding, rogue_label = extract(model, rogue_test, device)
    prototypes = np.stack([train_embedding[train_label == cls].mean(0) for cls in range(len(known))])
    prototypes /= np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8, None)
    val_score, _ = score(val_embedding, prototypes); known_score, known_prediction = score(known_embedding, prototypes)
    rogue_score, _ = score(rogue_embedding, prototypes)
    result_metrics = metrics(val_score, known_score, rogue_score, known_prediction == known_label, args.known_reject_rate)
    np.savez_compressed(output / "per_sample_scores.npz", validation_score=val_score, known_score=known_score,
                        known_label=known_label, known_prediction=known_prediction, rogue_score=rogue_score, rogue_label=rogue_label)
    result = {"method": "NS-RFF", "implementation": "official_cnn_l2softmax_protocol_adaptation_v1",
              "official_source": "https://github.com/xrj-com/NS-RFF", "split": args.split, "seed": args.seed,
              "known_classes": known, "rogue_classes": rogue, "best_epoch": saved["epoch"],
              "train_seconds": time.time() - start, "split_fingerprints": fingerprints, "metrics": result_metrics,
              "history": history, "protocol": {"architecture": "official CNN-L2Softmax, d2=24, 512D",
              "input_adapter": "deterministic leading 1280 complex samples required by official architecture",
              "objective": "official L2Softmax s=10 m=0", "score": "nearest class-prototype cosine distance",
              "checkpoint_selection": "minimum known-validation CE", "operating_threshold": "known-validation Q90",
              "target_rogue_used_for_fit_selection_or_threshold": False},
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(result, indent=2)); print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__": main()
