"""Train DenseNet-121 on the FULL NIH ChestX-ray14 (112,120 images) on Kaggle.

Self-contained: no imports from this repo, so it can be pasted into a Kaggle
notebook cell or run as a Kaggle script. Attach the dataset
`nih-chest-xrays/data` as input; nothing is downloaded to your own machine.

    python train_full_nih.py --img 224 --epochs 8 --max-hours 10.5

Design mirrors the local sample run so the two are comparable:
  * same architecture and head (DenseNet-121 -> GAP -> dropout 0.3 -> 14 logits)
  * same loss (weighted BCE, pos_weight capped at 20, label smoothing 0.05)
  * same two-stage schedule (head-only warm-up, then discriminative LRs + cosine)
  * same augmentation, same hflip TTA at test time
  * official NIH split: test_list.txt is never trained on, and train_val is
    split 90/10 by PATIENT for early stopping

Outputs (in /kaggle/working, download these):
    densenet121_full<img>.pt    best checkpoint, loadable by the local code
    full_test_preds.npz         {images, y_true, y_prob} on the 25,596 test films
    history.csv, summary.json
    last.pt                     resume state (re-run with --resume in a new session)

A first pass resizes every image to 256 px JPEG under /tmp (about 15 min with
4 workers); after that each epoch is fast. /tmp is not saved to the output.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
           "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
           "Pleural_Thickening", "Pneumonia", "Pneumothorax"]
IN = Path("/kaggle/input/datasets/nih-chest-xrays/data")
WORK = Path("/kaggle/working")
CACHE = Path("/tmp/cxr_cache")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------- data ---
def index_images() -> dict[str, Path]:
    paths = {}
    for p in IN.glob("images_*/images/*.png"):
        paths[p.name] = p
    if not paths:  # some mirrors flatten the folders
        for p in IN.rglob("*.png"):
            paths[p.name] = p
    return paths


def _cache_one(args):
    src, dst, px = args
    if dst.exists():
        return 0
    Image.open(src).convert("L").resize((px, px), Image.BILINEAR).save(dst, quality=92)
    return 1


def build_cache(paths: dict[str, Path], names: list[str], px: int, workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(paths[n], CACHE / (Path(n).stem + ".jpg"), px) for n in names if n in paths]
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_cache_one, jobs, chunksize=64), 1):
            done += r
            if i % 10000 == 0:
                print(f"  cache {i}/{len(jobs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"cache ready: {done} new, {len(jobs)} total, {(time.time() - t0) / 60:.1f} min")


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(IN / "Data_Entry_2017.csv")
    df.columns = [c.strip() for c in df.columns]
    labs = df["Finding Labels"].str.split("|")
    for c in CLASSES:
        df[c] = labs.apply(lambda ls, c=c: int(c in ls))
    return df[["Image Index", "Patient ID"] + CLASSES].rename(
        columns={"Image Index": "image", "Patient ID": "patient"})


def official_split(df: pd.DataFrame, seed: int):
    test_names = set((IN / "test_list.txt").read_text().split())
    tv_names = set((IN / "train_val_list.txt").read_text().split())
    test = df[df.image.isin(test_names)].reset_index(drop=True)
    tv = df[df.image.isin(tv_names)].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_idx, va_idx = next(gss.split(tv, groups=tv["patient"]))
    train = tv.iloc[tr_idx].reset_index(drop=True)
    val = tv.iloc[va_idx].reset_index(drop=True)
    assert not set(train.patient) & set(val.patient)
    assert not set(tv.patient) & set(test.patient), "official split leaks patients"
    return train, val, test


def transforms(px: int, train: bool) -> T.Compose:
    if train:
        geom = [T.RandomResizedCrop(px, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                T.ColorJitter(brightness=0.15, contrast=0.15)]
    else:
        geom = [T.Resize(px), T.CenterCrop(px)]
    return T.Compose(geom + [T.Grayscale(3), T.ToTensor(), T.Normalize(MEAN, STD)])


class CXR(Dataset):
    def __init__(self, df: pd.DataFrame, tf: T.Compose):
        self.names = df["image"].tolist()
        self.y = df[CLASSES].to_numpy(dtype=np.float32)
        self.tf = tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(CACHE / (Path(self.names[i]).stem + ".jpg")).convert("L")
        return self.tf(img), torch.from_numpy(self.y[i])


# ---------------------------------------------------------------- model ---
class CXRNet(nn.Module):
    """Identical module names to the local src/model.py so state_dicts interchange."""

    def __init__(self, n_classes: int = 14, dropout: float = 0.3):
        super().__init__()
        net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.classifier(self.drop(self.pool(x).flatten(1)))

    def freeze_backbone(self, frozen: bool):
        for p in self.features.parameters():
            p.requires_grad = not frozen


def macro_auroc(y, p):
    vals = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])
            if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(vals))


@torch.no_grad()
def predict(model, dl, device, tta: bool):
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float())) / 2
        ps.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------- train ---
@torch.no_grad()
def validate_loss(model, dl, device, criterion):
    model.eval()
    total, n = 0.0, 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=device.type == "cuda"):
            loss = criterion(model(x), y)

        total += loss.item() * len(x)
        n += len(x)

    return total / n
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--cache-px", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8, help="total incl. 1 head-only epoch")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train images")
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    paths = index_images()
    df = load_labels()
    df = df[df.image.isin(paths)].reset_index(drop=True)
    train, val, test = official_split(df, args.seed)
    if args.limit:
        train = train.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    print(f"images indexed {len(paths):,} | train {len(train):,}  val {len(val):,}  "
          f"test {len(test):,}")
    build_cache(paths, pd.concat([train, val, test]).image.tolist(), args.cache_px, args.workers)

    def dl(d, tf, shuffle):
        return DataLoader(CXR(d, tf), batch_size=args.bs, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=True, drop_last=shuffle)

    train_dl = dl(train, transforms(args.img, True), True)
    val_dl = dl(val, transforms(args.img, False), False)
    test_dl = dl(test, transforms(args.img, False), False)

    model = CXRNet().to(device)
    pos = train[CLASSES].sum().to_numpy(dtype=np.float32)
    pw = torch.from_numpy(np.clip((len(train) - pos) / np.maximum(pos, 1), 1, 20)
                          .astype(np.float32)).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def crit(z, y):
        return bce(z, y * 0.95 + 0.025)  # same label smoothing as the local run

    scaler = torch.amp.GradScaler("cuda")

    def make_opt(stage):
        if stage == 1:
            model.freeze_backbone(True)
            return torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        model.freeze_backbone(False)
        return torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": args.lr_backbone},
            {"params": model.classifier.parameters(), "lr": args.lr_head},
        ], weight_decay=1e-4)

    start_epoch, best, history = 1, -1.0, []
    patience = 3
    epochs_without_improvement = 0
    opt, sched = None, None
    ckpt_best = WORK / f"densenet121_full{args.img}.pt"
    last = WORK / "last.pt"
    if args.resume and last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        start_epoch, best, history = st["epoch"] + 1, st["best"], st["history"]
        print(f"resumed at epoch {start_epoch}, best {best:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        stage = 1 if epoch == 1 else 2
        if opt is None or epoch == 2:
            opt = make_opt(stage)
            sched = None
            if stage == 2:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - 1)
                for _ in range(epoch - 2):  # fast-forward after a resume
                    sched.step()

        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
            n += len(x)
            if i % 200 == 0:
                print(f"  ep{epoch} it{i}/{len(train_dl)} loss {tot / n:.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if sched is not None:
            sched.step()

        val_loss = validate_loss(model, val_dl, device, crit)

        yv, pv = predict(model, val_dl, device, tta=False)
        auroc = macro_auroc(yv, pv)
        history.append({"stage": stage, "epoch": epoch, "train_loss": tot / n,
                        "val_loss": val_loss, "val_auroc": auroc,
                        "minutes": (time.time() - t0) / 60})
        flag = ""
        if auroc > best:
            best = auroc
            flag = "  *"
            epochs_without_improvement = 0

            torch.save({
                 "model": model.state_dict(),
                 "epoch": epoch,
                 "auroc": auroc
            }, WORK / "best.pt")

        else:
            epochs_without_improvement += 1



        print(f"epochs without improvement: {epochs_without_improvement}")

        print(
            f"epoch {epoch} stage {stage}  train {tot / n:.4f}  "
            f"val AUROC {auroc:.4f}{flag}  "
            f"({(time.time() - t0) / 60:.1f} min)",
            flush=True
        )

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best": best,
            "history": history
        }, last)

        pd.DataFrame(history).to_csv(WORK / "history.csv", index=False)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Time budget
        elapsed_h = (time.time() - t_start) / 3600
        per_epoch_h = (time.time() - t0) / 3600

        if epoch < args.epochs and elapsed_h + per_epoch_h * 1.3 > args.max_hours:
            print(
                f"stopping: {elapsed_h:.1f} h used, "
                f"next epoch would exceed budget"
            )
            break

    # ---- final test evaluation with the best checkpoint --------------------
    state = torch.load(ckpt_best, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])

    yt, pt = predict(model, test_dl, device, tta=True)

    per_class = {
        c: float(roc_auc_score(yt[:, i], pt[:, i]))
        for i, c in enumerate(CLASSES)
        if 0 < yt[:, i].sum() < len(yt)
    }

    summary = {
        "model": f"densenet121_full{args.img}",
        "img": args.img,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "best_val_auroc": best,
        "test_macro_auroc_tta": float(np.mean(list(per_class.values()))),
        "test_per_class_auroc": per_class,
        "epochs_run": len(history),
        "hours": (time.time() - t_start) / 3600
    }

    np.savez(
        WORK / "full_test_preds.npz",
        images=np.array(test.image.tolist()),
        y_true=yt,
        y_prob=pt
    )

    (WORK / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(
        {k: v for k, v in summary.items() if k != "test_per_class_auroc"},
        indent=2
    ))

    print(
        "test per-class AUROC:",
        {k: round(v, 3) for k, v in per_class.items()}
    )

    print(
        "done. download: densenet121_full*.pt, "
        "full_test_preds.npz, history.csv, summary.json"
    )


if __name__ == "__main__":
    main()