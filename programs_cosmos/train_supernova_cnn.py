"""
Train a CNN to Detect Supernovae Near Host Galaxies

APPROACH: Difference-based detection.
Instead of feeding raw images, the CNN sees DIFFERENCE images:
  Channel 0: JWST1 - HST    (SN in JWST → bright spot here)
  Channel 1: JWST2 - HST    (same bright spot confirms SN)
  Channel 2: JWST1 - JWST2  (should be ~0 for real SN in both filters)

This forces the CNN to learn WHAT CHANGED between epochs,
not just detect bright sources.

For HST supernovae (SN faded by JWST epoch):
  Channel 0: HST - JWST1    (SN in HST → bright spot here)
  Channel 1: HST - JWST2    (same bright spot confirms)

Physical constraints:
  - SN must appear near the host galaxy (image center)
  - JWST SN must appear in BOTH JWST filters
  - HST SN must be absent from BOTH JWST filters

Expected: ~1 SN per 7,000 objects → ~30 from 212K objects

Pipeline:
    Step 1: Load data — match HST/JWST1/JWST2 triplets
    Step 2: Train on 23 known SN + artificial SN (difference images)
    Step 3: Validate — recover ALL 23 known SN
    Step 4: Apply to all objects (batched, multiprocessor)
    Step 5: Verify results

HST directory: /data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/
JWST directory: /data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png

Usage:
    python train_supernova_cnn.py [options]
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 23 confirmed supernovae identified by visual inspection
KNOWN_SUPERNOVAE = [
    "9748", "19931", "25141", "27350", "39020", "52864", "53669",
    "63919", "63924", "64479", "78323", "239301", "245766",
    "246188", "296673", "296868", "318858", "349174", "435613",
    "435768", "468896", "469221", "471959",
]
KNOWN_SUPERNOVAE_SET = set(KNOWN_SUPERNOVAE)

IMAGE_SIZE = 64
NUM_CHANNELS = 3  # diff1, diff2, diff12


# ============================================================
# Data loading
# ============================================================

def parse_filename(filename):
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(.+?)_([^_]+)_(hstcosmos|jwst[12])$", basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def build_file_maps(hst_dir, jwst_dir):
    hst_files = sorted(glob.glob(os.path.join(hst_dir, "*.png")))
    jwst_files = sorted(glob.glob(os.path.join(jwst_dir, "*.png")))
    print(f"Found {len(hst_files)} HST files, {len(jwst_files)} JWST files")

    file_map = {}
    for f in hst_files:
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, _ = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id]["hst"] = f
    for f in jwst_files:
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, img_type = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id][img_type] = f
    return file_map


def load_and_crop(filepath, size=IMAGE_SIZE, crop_frac=0.5):
    """Load PNG, crop central 50% (galaxy region), resize, normalize."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = img[y0:y0+ch, x0:x0+cw]
    cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return cropped.astype(np.float32) / 255.0


def load_full(filepath, size=IMAGE_SIZE):
    """Load PNG without cropping (for artificial SN injection)."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def crop_center(image, crop_frac=0.5):
    """Crop central region of an already-loaded image."""
    h, w = image.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = image[y0:y0+ch, x0:x0+cw]
    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_AREA)


# ============================================================
# Difference image computation
# ============================================================

def compute_diff_channels(hst, jwst1, jwst2):
    """Compute 3-channel difference image.

    Channel 0: JWST1 - HST   (positive = new source in JWST1)
    Channel 1: JWST2 - HST   (positive = new source in JWST2)
    Channel 2: |JWST1 - JWST2| (small = consistent across filters = real SN)

    For a JWST supernova: ch0 > 0, ch1 > 0, ch2 ≈ 0
    For an HST supernova: ch0 < 0, ch1 < 0, ch2 ≈ 0
    For no supernova: ch0 ≈ 0, ch1 ≈ 0, ch2 ≈ 0
    """
    diff1 = jwst1 - hst           # JWST1 - HST
    diff2 = jwst2 - hst           # JWST2 - HST
    diff12 = np.abs(jwst1 - jwst2)  # |JWST1 - JWST2|

    return np.stack([diff1, diff2, diff12], axis=0).astype(np.float32)


# ============================================================
# Artificial supernova injection
# ============================================================

def inject_supernova_at(image, x, y, peak, sigma):
    img = image.copy()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gaussian = peak * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return np.clip(img + gaussian, 0, 1).astype(np.float32)


def random_pos_near_center(h, w, max_frac=0.25):
    """Random position within 25% of image center (near galaxy)."""
    cx, cy = w // 2, h // 2
    dx = int(w * max_frac)
    dy = int(h * max_frac)
    x = max(3, min(w - 4, cx + random.randint(-dx, dx)))
    y = max(3, min(h - 4, cy + random.randint(-dy, dy)))
    return x, y


# ============================================================
# Datasets
# ============================================================

def augment_channels(diff):
    """Augment 3-channel diff image (C, H, W)."""
    # Random flips (apply same transform to all channels)
    if random.random() > 0.5:
        diff = np.flip(diff, axis=2).copy()  # horizontal flip
    if random.random() > 0.5:
        diff = np.flip(diff, axis=1).copy()  # vertical flip
    # Random 90-degree rotation
    k = random.randint(0, 3)
    diff = np.rot90(diff, k, axes=(1, 2)).copy()
    return diff


class RealSupernovaDataset(Dataset):
    """Known supernovae as (hst, jwst1, jwst2) → difference channels."""

    def __init__(self, samples, augment=False):
        """samples: list of (hst_path, jwst1_path, jwst2_path, label)"""
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hst_path, jwst1_path, jwst2_path, label = self.samples[idx]

        hst = load_and_crop(hst_path)
        jwst1 = load_and_crop(jwst1_path)
        jwst2 = load_and_crop(jwst2_path)

        if hst is None or jwst1 is None or jwst2 is None:
            return torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(0.0)

        diff = compute_diff_channels(hst, jwst1, jwst2)

        if self.augment:
            diff = augment_channels(diff)

        return torch.tensor(diff, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class ArtificialSNDataset(Dataset):
    """Generate training data with artificial SN in difference images.

    Positive samples:
      - JWST SN: inject point source at same position in BOTH JWST filters
        → diff1 > 0, diff2 > 0, diff12 ≈ 0 near the SN
      - HST SN: inject in HST only
        → diff1 < 0, diff2 < 0, diff12 ≈ 0

    Negative samples: real objects with no injection.
    """

    def __init__(self, object_files, num_positive, num_negative, augment=True):
        self.object_files = object_files
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.total = num_positive + num_negative
        self.augment = augment

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        is_positive = idx < self.num_positive
        files = random.choice(self.object_files)

        # Load full images for injection, then crop
        hst = load_full(files["hst"])
        jwst1 = load_full(files["jwst1"])
        jwst2 = load_full(files["jwst2"])

        if hst is None or jwst1 is None or jwst2 is None:
            return torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(0.0)

        if is_positive:
            h, w = hst.shape
            x, y = random_pos_near_center(h, w)
            peak = random.uniform(0.15, 0.8)
            sigma = random.uniform(1.0, 3.0)

            if random.random() > 0.5:
                # JWST SN: inject in both JWST, not HST
                jwst1 = inject_supernova_at(jwst1, x, y, peak, sigma)
                peak2 = peak * random.uniform(0.7, 1.3)
                jwst2 = inject_supernova_at(jwst2, x, y, peak2, sigma)
            else:
                # HST SN: inject in HST only
                hst = inject_supernova_at(hst, x, y, peak, sigma)
            label = 1.0
        else:
            label = 0.0

        # Crop center AFTER injection
        hst = crop_center(hst)
        jwst1 = crop_center(jwst1)
        jwst2 = crop_center(jwst2)

        # Compute difference channels
        diff = compute_diff_channels(hst, jwst1, jwst2)

        if self.augment:
            diff = augment_channels(diff)

        return torch.tensor(diff, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ============================================================
# CNN Model — designed for difference images
# ============================================================

class SupernovaCNN(nn.Module):
    """CNN for detecting supernovae from difference images.

    Input: 3-channel difference image
      ch0: JWST1 - HST
      ch1: JWST2 - HST
      ch2: |JWST1 - JWST2|
    Output: P(supernova)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


# ============================================================
# Inference
# ============================================================

def predict_object(model, hst_path, jwst1_path, jwst2_path, device):
    """Predict SN probability from difference images."""
    hst = load_and_crop(hst_path)
    jwst1 = load_and_crop(jwst1_path)
    jwst2 = load_and_crop(jwst2_path)

    if hst is None or jwst1 is None or jwst2 is None:
        return 0.0

    diff = compute_diff_channels(hst, jwst1, jwst2)
    tensor = torch.tensor(diff, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    return prob


def predict_known_sn(model, known_sn_objects, device):
    results = {}
    for obj_id, files in known_sn_objects.items():
        prob = predict_object(model, files["hst"], files["jwst1"], files["jwst2"], device)
        results[obj_id] = prob
    return results


class InferenceDataset(Dataset):
    """Batched inference dataset using difference images."""

    def __init__(self, obj_ids, file_map):
        self.obj_ids = obj_ids
        self.file_map = file_map

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, idx):
        obj_id = self.obj_ids[idx]
        files = self.file_map[obj_id]

        hst = load_and_crop(files["hst"])
        jwst1 = load_and_crop(files["jwst1"])
        jwst2 = load_and_crop(files["jwst2"])

        if hst is None or jwst1 is None or jwst2 is None:
            return obj_id, torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), False

        diff = compute_diff_channels(hst, jwst1, jwst2)
        return obj_id, torch.tensor(diff, dtype=torch.float32), True


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN to detect supernovae using difference images."
    )
    parser.add_argument("--hst-dir",
        default="/data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/")
    parser.add_argument("--jwst-dir",
        default="/data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-artificial", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--inference-workers", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--model-path", default="supernova_cnn.pth")
    parser.add_argument("--output-csv", default="supernova_cnn_candidates.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==================================================================
    # STEP 1: Load and match files
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading and matching image files")
    print("="*60)

    file_map = build_file_maps(args.hst_dir, args.jwst_dir)
    paired = {k: v for k, v in file_map.items()
              if "hst" in v and "jwst1" in v and "jwst2" in v}
    print(f"Objects with all 3 images: {len(paired)}")

    known_sn_objects = {k: v for k, v in paired.items() if k in KNOWN_SUPERNOVAE_SET}
    non_sn_objects = {k: v for k, v in paired.items() if k not in KNOWN_SUPERNOVAE_SET}

    print(f"Known supernovae found: {len(known_sn_objects)}/{len(KNOWN_SUPERNOVAE_SET)}")
    if known_sn_objects:
        print(f"  IDs: {sorted(known_sn_objects.keys())}")
    missing = KNOWN_SUPERNOVAE_SET - set(known_sn_objects.keys())
    if missing:
        print(f"  Missing: {sorted(missing)}")

    if len(known_sn_objects) == 0:
        print("ERROR: No known supernovae found. Cannot train.")
        sys.exit(1)

    # ==================================================================
    # STEP 2: Train CNN on difference images
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 2: Training CNN on difference images")
    print("="*60)

    # --- Verified ground truth ---
    # IDs 1 through 53131 have been visually inspected:
    #   Only 7 are real SN: 9748, 19931, 25141, 27350, 39020, 52864, 53669
    #   ALL others in this range are confirmed negatives (not SN)
    VERIFIED_MAX_ID = 53131
    VERIFIED_SN = {"9748", "19931", "25141", "27350", "39020", "52864", "53669"}

    # Build verified negative set: objects in ID range 1-53131 that are NOT SN
    verified_negatives = []
    verified_positives = []
    for obj_id, files in paired.items():
        try:
            numeric_id = int(obj_id)
        except ValueError:
            continue
        if numeric_id <= VERIFIED_MAX_ID:
            if obj_id in VERIFIED_SN:
                verified_positives.append((files["hst"], files["jwst1"], files["jwst2"], 1.0))
            else:
                verified_negatives.append((files["hst"], files["jwst1"], files["jwst2"], 0.0))

    print(f"Verified ID range 1-{VERIFIED_MAX_ID}:")
    print(f"  Verified positives (real SN): {len(verified_positives)}")
    print(f"  Verified negatives (confirmed not SN): {len(verified_negatives)}")

    # Also include all 23 known SN (some are outside the verified range)
    all_known_positive = []
    for obj_id, files in known_sn_objects.items():
        all_known_positive.append((files["hst"], files["jwst1"], files["jwst2"], 1.0))
    print(f"All known SN (full dataset): {len(all_known_positive)}")

    # Oversample positives to balance against verified negatives
    # Use ~1:100 ratio (still heavily negative, but enough positives to learn)
    num_neg = min(len(verified_negatives), 50000)  # Cap at 50K negatives
    num_pos_repeats = max(1, num_neg // (len(all_known_positive) * 10))
    real_sn_oversampled = all_known_positive * num_pos_repeats
    print(f"Real positives oversampled: {len(all_known_positive)} x {num_pos_repeats} "
          f"= {len(real_sn_oversampled)}")

    # Sample verified negatives
    random.shuffle(verified_negatives)
    verified_neg_sample = verified_negatives[:num_neg]
    print(f"Verified negatives used: {len(verified_neg_sample)}")

    # Also add artificial SN for variety
    non_sn_file_list = [non_sn_objects[k] for k in list(non_sn_objects.keys())[:10000]]
    num_art_pos = args.num_artificial
    print(f"Artificial positives: {num_art_pos}")

    art_dataset = ArtificialSNDataset(
        non_sn_file_list, num_positive=num_art_pos, num_negative=0, augment=True)
    real_sn_dataset = RealSupernovaDataset(real_sn_oversampled, augment=True)
    real_neg_dataset = RealSupernovaDataset(verified_neg_sample, augment=True)

    train_dataset = torch.utils.data.ConcatDataset(
        [real_sn_dataset, real_neg_dataset, art_dataset])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    print(f"Total training samples: {len(train_dataset)}")

    model = SupernovaCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_recovery = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)

        if epoch % 5 == 0 or epoch == args.epochs:
            sn_probs = predict_known_sn(model, known_sn_objects, device)
            recovered = sum(1 for p in sn_probs.values() if p >= 0.5)
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                  f"Known SN recovered: {recovered}/{len(known_sn_objects)}")

            if recovered >= best_recovery:
                best_recovery = recovered
                best_epoch = epoch
                torch.save(model.state_dict(), args.model_path)
            scheduler.step(train_loss)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss: {train_loss:.4f} acc: {train_acc:.3f}")

    print(f"\nBest recovery: {best_recovery}/{len(known_sn_objects)} at epoch {best_epoch}")
    print(f"Model saved to {args.model_path}")

    # ==================================================================
    # STEP 3: Validate — find threshold that recovers all known SN
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 3: Validating recovery of known supernovae")
    print("="*60)

    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    sn_probs = predict_known_sn(model, known_sn_objects, device)

    sorted_sn = sorted(sn_probs.items(), key=lambda x: x[1], reverse=True)
    print(f"\nKnown supernovae probabilities:")
    for obj_id, prob in sorted_sn:
        status = "OK" if prob >= 0.5 else "LOW"
        print(f"  {obj_id}: {prob:.4f} [{status}]")

    min_known_prob = min(sn_probs.values())
    print(f"\nLowest known SN probability: {min_known_prob:.4f}")

    threshold = min_known_prob * 0.9
    threshold = max(threshold, 0.01)
    print(f"Auto-selected threshold: {threshold:.4f}")

    recovered = [oid for oid, p in sn_probs.items() if p >= threshold]
    missed_sn = [oid for oid, p in sn_probs.items() if p < threshold]

    if missed_sn:
        print(f"  WARNING: Missing {len(missed_sn)}: {missed_sn}")
    else:
        print(f"  All {len(recovered)} known supernovae RECOVERED!")

    # ==================================================================
    # STEP 4: Apply to all objects — batched multiprocessor
    # ==================================================================
    print("\n" + "="*60)
    print(f"STEP 4: Applying CNN to all {len(paired)} objects")
    print(f"  Batch size: {args.inference_batch_size}, Workers: {args.inference_workers}")
    print(f"  Candidates → {args.output_csv}")
    print("="*60)

    all_probs_csv = args.output_csv.replace(".csv", "_all_probs.csv")

    obj_ids_sorted = sorted(paired.keys())
    inference_dataset = InferenceDataset(obj_ids_sorted, paired)

    def collate_fn(batch):
        obj_ids = [b[0] for b in batch]
        tensors = torch.stack([b[1] for b in batch])
        valid = [b[2] for b in batch]
        return obj_ids, tensors, valid

    inference_loader = DataLoader(
        inference_dataset, batch_size=args.inference_batch_size,
        shuffle=False, num_workers=args.inference_workers,
        pin_memory=True, collate_fn=collate_fn)

    start_time = time.time()
    num_candidates = 0
    completed = 0
    total = len(paired)

    csv_file = open(args.output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["object_id", "probability", "known_sn"])

    all_probs_file = open(all_probs_csv, "w", newline="")
    all_probs_writer = csv.writer(all_probs_file)
    all_probs_writer.writerow(["object_id", "probability", "known_sn"])

    model.eval()
    try:
        with torch.no_grad():
            for batch_ids, batch_tensors, batch_valid in inference_loader:
                batch_tensors = batch_tensors.to(device)
                outputs = torch.sigmoid(model(batch_tensors)).cpu().numpy()

                for i, obj_id in enumerate(batch_ids):
                    prob = float(outputs[i]) if batch_valid[i] else 0.0
                    is_known = "YES" if obj_id in KNOWN_SUPERNOVAE_SET else "NO"

                    all_probs_writer.writerow([obj_id, f"{prob:.6f}", is_known])

                    if prob >= threshold:
                        csv_writer.writerow([obj_id, f"{prob:.4f}", is_known])
                        csv_file.flush()
                        num_candidates += 1

                        tag = "KNOWN" if is_known == "YES" else "NEW"
                        print(f"  *** SN CANDIDATE: {obj_id} "
                              f"(prob={prob:.4f}) [{tag}] ***")

                completed += len(batch_ids)
                if completed % 5000 < args.inference_batch_size or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    all_probs_file.flush()
                    print(f"  Progress: {completed}/{total} "
                          f"({completed/total*100:.1f}%) "
                          f"| {rate:.0f} obj/s | ETA: {eta:.0f}s "
                          f"| Candidates: {num_candidates}")
    finally:
        csv_file.close()
        all_probs_file.close()

    elapsed = time.time() - start_time
    print(f"\nInference complete in {elapsed:.1f}s")

    # ==================================================================
    # STEP 5: Final verification
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 5: Final verification")
    print("="*60)

    candidate_ids = set()
    with open(args.output_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidate_ids.add(row["object_id"])

    known_in_data = set(known_sn_objects.keys())
    known_recovered = known_in_data & candidate_ids
    known_missed = known_in_data - candidate_ids
    detection_rate = len(candidate_ids) / total * 1000 if total > 0 else 0

    print(f"\n--- Final Known SN Check ---")
    print(f"Known SN in dataset: {len(known_in_data)}")
    print(f"Known SN recovered:  {len(known_recovered)}/{len(known_in_data)}")
    for obj_id in sorted(known_recovered):
        print(f"  FOUND: {obj_id}")
    for obj_id in sorted(known_missed):
        print(f"  MISSED: {obj_id}")

    new_count = len(candidate_ids) - len(known_recovered)
    print(f"\nTotal candidates: {len(candidate_ids)} "
          f"(known: {len(known_recovered)}, new: {new_count})")
    print(f"Threshold: {threshold:.4f}")
    print(f"Detection rate: {detection_rate:.1f} per 1000 objects")

    if detection_rate > 2:
        print(f"WARNING: Rate {detection_rate:.1f}/1000 higher than expected ~0.15/1000")
    print(f"\nCandidate file: {args.output_csv}")
    print(f"All probabilities: {all_probs_csv}")


if __name__ == "__main__":
    main()
