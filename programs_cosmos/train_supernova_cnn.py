"""
Train a CNN to Detect Supernovae Near Host Galaxies

Each PNG has a galaxy at the center. Supernovae appear as new point
sources near the host galaxy. The CNN focuses on the central galaxy
region by cropping the central 50% of each image before analysis.

Pipeline:
    Step 1: Load data — match HST/JWST1/JWST2 triplets by object ID
    Step 2: Train — use 23 known SN + artificial SN (placed near galaxy)
            as positives, real non-SN objects as negatives
    Step 3: Validate — verify ALL 23 known SN are recovered.
            If not, lower threshold until all are found.
    Step 4: Apply — run trained model on entire dataset
    Step 5: Verify — confirm known SN are in final results,
            check detection rate is ~1 per 1000

Expected: ~212 candidates from ~212K objects (1 in 1000)

HST directory: /data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/
JWST directory: /data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png

Usage:
    python train_supernova_cnn.py [options]
    python train_supernova_cnn.py --epochs 50 --num-workers 8
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
    "471959", "53669", "296673", "296868", "318858", "320233",
    "9748", "52864", "407613", "239246", "239875",
    "245766", "468896", "469221", "120035", "63919", "78323",
    "25141", "25283", "27350", "435613", "435768",
]
KNOWN_SUPERNOVAE_SET = set(KNOWN_SUPERNOVAE)

IMAGE_SIZE = 64
NUM_CHANNELS = 3  # HST, JWST1, JWST2


# ============================================================
# Data loading and file matching
# ============================================================

def parse_filename(filename):
    """Extract ID and Field from a filename."""
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(.+?)_([^_]+)_(hstcosmos|jwst[12])$", basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def build_file_maps(hst_dir, jwst_dir):
    """Build mapping from object ID to HST and JWST file paths."""
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


def load_and_resize(filepath, size=IMAGE_SIZE):
    """Load a PNG, convert to grayscale, resize, normalize to [0, 1]."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def crop_center(image, crop_frac=0.5):
    """Crop the central region of an image.

    The galaxy is at the center of each PNG. Cropping focuses the CNN
    on the galaxy and its immediate surroundings, removing irrelevant
    background at the edges.

    Args:
        image: 2D float32 array
        crop_frac: fraction of image to keep (0.5 = central 50%)

    Returns:
        Cropped image resized back to IMAGE_SIZE.
    """
    h, w = image.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = image[y0:y0+ch, x0:x0+cw]
    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_AREA)


def load_and_crop_center(filepath, size=IMAGE_SIZE, crop_frac=0.5):
    """Load PNG, crop central galaxy region, resize, normalize."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Crop center first (galaxy region), then resize
    h, w = img.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = img[y0:y0+ch, x0:x0+cw]
    cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return cropped.astype(np.float32) / 255.0


# ============================================================
# Artificial supernova injection
# ============================================================

def inject_supernova_at(image, x, y, peak, sigma):
    """Inject a Gaussian point source at (x, y) into the image."""
    img = image.copy()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gaussian = peak * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return np.clip(img + gaussian, 0, 1).astype(np.float32)


def random_position_near_center(h, w, max_offset_frac=0.25):
    """Generate a random position near the image center (host galaxy).

    After center-cropping, the galaxy fills most of the image.
    SN is placed within 25% of center — right on or next to the galaxy.
    """
    cx, cy = w // 2, h // 2
    max_dx = int(w * max_offset_frac)
    max_dy = int(h * max_offset_frac)
    x = cx + random.randint(-max_dx, max_dx)
    y = cy + random.randint(-max_dy, max_dy)
    x = max(3, min(w - 4, x))
    y = max(3, min(h - 4, y))
    return x, y


def add_artificial_supernova(image):
    """Inject one artificial supernova near the host galaxy (image center)."""
    h, w = image.shape
    x, y = random_position_near_center(h, w)
    peak = random.uniform(0.3, 0.9)
    sigma = random.uniform(1.0, 3.0)
    return inject_supernova_at(image, x, y, peak, sigma), (x, y, peak, sigma)


# ============================================================
# Datasets
# ============================================================

def augment_images(images):
    """Apply random flips and rotation to a list of images consistently."""
    if random.random() > 0.5:
        images = [np.fliplr(img).copy() for img in images]
    if random.random() > 0.5:
        images = [np.flipud(img).copy() for img in images]
    k = random.randint(0, 3)
    images = [np.rot90(img, k).copy() for img in images]
    return images


class SupernovaDataset(Dataset):
    """Dataset of (HST, JWST1, JWST2) image triplets with labels."""

    def __init__(self, samples, augment=False):
        """samples: list of (hst_path, jwst1_path, jwst2_path, label)"""
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hst_path, jwst1_path, jwst2_path, label = self.samples[idx]

        hst_img = load_and_crop_center(hst_path)
        jwst1_img = load_and_crop_center(jwst1_path)
        jwst2_img = load_and_crop_center(jwst2_path)

        if hst_img is None or jwst1_img is None or jwst2_img is None:
            return torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(0.0)

        if self.augment:
            hst_img, jwst1_img, jwst2_img = augment_images(
                [hst_img, jwst1_img, jwst2_img])

        triplet = np.stack([hst_img, jwst1_img, jwst2_img], axis=0)
        return (torch.tensor(triplet, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


class ArtificialSupernovaDataset(Dataset):
    """Generate training data with artificial supernovae.

    Physical constraints enforced:
    - JWST SN: injected at SAME position in BOTH jwst1 AND jwst2, NOT in HST
    - HST SN: injected in HST ONLY, not in jwst1 or jwst2
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
        hst_img = load_and_resize(files["hst"])
        jwst1_img = load_and_resize(files["jwst1"])
        jwst2_img = load_and_resize(files["jwst2"])

        if hst_img is None or jwst1_img is None or jwst2_img is None:
            return torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(0.0)

        if is_positive:
            if random.random() > 0.5:
                # JWST SN: same source in both JWST filters, not in HST
                # Place near center (where the host galaxy is)
                h, w = jwst1_img.shape
                x, y = random_position_near_center(h, w)
                peak = random.uniform(0.3, 0.9)
                sigma = random.uniform(1.0, 3.0)
                jwst1_img = inject_supernova_at(jwst1_img, x, y, peak, sigma)
                # Slightly different brightness in filter 2 (realistic)
                peak2 = peak * random.uniform(0.7, 1.3)
                jwst2_img = inject_supernova_at(jwst2_img, x, y, peak2, sigma)
            else:
                # HST SN: only in HST, near center (host galaxy)
                hst_img, _ = add_artificial_supernova(hst_img)
            label = 1.0
        else:
            label = 0.0

        if self.augment:
            hst_img, jwst1_img, jwst2_img = augment_images(
                [hst_img, jwst1_img, jwst2_img])

        triplet = np.stack([hst_img, jwst1_img, jwst2_img], axis=0)
        return (torch.tensor(triplet, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


# ============================================================
# CNN Model
# ============================================================

class SupernovaCNN(nn.Module):
    """3-channel CNN: HST + JWST1 + JWST2 -> supernova probability."""

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
# Training and inference
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


def predict_object(model, hst_path, jwst1_path, jwst2_path, device):
    """Predict supernova probability for one (HST, JWST1, JWST2) triplet."""
    hst_img = load_and_resize(hst_path)
    jwst1_img = load_and_resize(jwst1_path)
    jwst2_img = load_and_resize(jwst2_path)

    if hst_img is None or jwst1_img is None or jwst2_img is None:
        return 0.0

    triplet = np.stack([hst_img, jwst1_img, jwst2_img], axis=0)
    tensor = torch.tensor(triplet, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    return prob


def predict_known_sn(model, known_sn_objects, device):
    """Predict probabilities for all known supernovae. Returns dict {id: prob}."""
    results = {}
    for obj_id, files in known_sn_objects.items():
        prob = predict_object(model, files["hst"], files["jwst1"], files["jwst2"], device)
        results[obj_id] = prob
    return results


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN to detect supernovae in JWST/HST image triplets."
    )
    parser.add_argument(
        "--hst-dir",
        default="/data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/",
        help="Directory containing HST PNG images"
    )
    parser.add_argument(
        "--jwst-dir",
        default="/data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png",
        help="Directory containing JWST PNG images"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--num-artificial", type=int, default=5000,
                        help="Number of artificial SN positive samples (default: 5000)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--model-path", default="supernova_cnn.pth",
                        help="Path to save trained model")
    parser.add_argument("--output-csv", default="supernova_cnn_candidates.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==================================================================
    # STEP 1: Build file maps — require all 3 images per object
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading and matching image files")
    print("="*60)

    file_map = build_file_maps(args.hst_dir, args.jwst_dir)
    paired = {k: v for k, v in file_map.items()
              if "hst" in v and "jwst1" in v and "jwst2" in v}
    print(f"Objects with all 3 images (HST+JWST1+JWST2): {len(paired)}")

    known_sn_objects = {k: v for k, v in paired.items() if k in KNOWN_SUPERNOVAE_SET}
    non_sn_objects = {k: v for k, v in paired.items() if k not in KNOWN_SUPERNOVAE_SET}

    print(f"Known supernovae found: {len(known_sn_objects)}/{len(KNOWN_SUPERNOVAE_SET)}")
    if known_sn_objects:
        print(f"  Found IDs: {sorted(known_sn_objects.keys())}")
    missing = KNOWN_SUPERNOVAE_SET - set(known_sn_objects.keys())
    if missing:
        print(f"  Missing IDs (no triplet): {sorted(missing)}")

    if len(known_sn_objects) == 0:
        print("ERROR: No known supernovae found in dataset. Cannot train.")
        sys.exit(1)

    # ==================================================================
    # STEP 2: Train CNN
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 2: Training CNN")
    print("="*60)

    # Real positive samples: known supernovae (oversampled heavily)
    real_positive = []
    for obj_id, files in known_sn_objects.items():
        real_positive.append((files["hst"], files["jwst1"], files["jwst2"], 1.0))

    # Oversample known SN: each known SN appears 50 times in training
    real_sn_oversampled = real_positive * 50
    print(f"Real positive samples: {len(real_positive)} x 50 = {len(real_sn_oversampled)}")

    # Non-SN objects for artificial generation pool
    non_sn_keys = list(non_sn_objects.keys())
    random.shuffle(non_sn_keys)
    non_sn_file_list = [non_sn_objects[k] for k in non_sn_keys[:10000]]

    # Artificial data: match ~1/1000 rate
    # Use 1000:1 negative:positive ratio in artificial data
    num_art_pos = args.num_artificial
    num_art_neg = num_art_pos * 20
    print(f"Artificial data: {num_art_pos} positive + {num_art_neg} negative")

    art_dataset = ArtificialSupernovaDataset(
        non_sn_file_list,
        num_positive=num_art_pos,
        num_negative=num_art_neg,
        augment=True,
    )
    real_sn_dataset = SupernovaDataset(real_sn_oversampled, augment=True)

    train_dataset = torch.utils.data.ConcatDataset([art_dataset, real_sn_dataset])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Total training samples: {len(train_dataset)}")

    # Train
    model = SupernovaCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_recovery = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        train_loss = total_loss / total
        train_acc = correct / total

        # Check recovery of known SN every 5 epochs
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
    # STEP 3: Validate — ensure ALL known SN are recovered
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 3: Validating recovery of known supernovae")
    print("="*60)

    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    sn_probs = predict_known_sn(model, known_sn_objects, device)

    # Sort by probability to show all results
    sorted_sn = sorted(sn_probs.items(), key=lambda x: x[1], reverse=True)
    print(f"\nKnown supernovae probabilities:")
    for obj_id, prob in sorted_sn:
        status = "OK" if prob >= 0.5 else "LOW"
        print(f"  {obj_id}: {prob:.4f} [{status}]")

    # Find optimal threshold that recovers all known SN
    min_known_prob = min(sn_probs.values())
    print(f"\nLowest known SN probability: {min_known_prob:.4f}")

    # Set threshold just below the lowest known SN probability
    # This ensures ALL known SN are recovered
    threshold = min_known_prob * 0.9  # 10% margin below lowest known SN
    threshold = max(threshold, 0.01)  # Never go below 0.01
    print(f"Auto-selected threshold: {threshold:.4f}")

    # Verify all known SN pass the threshold
    recovered = [obj_id for obj_id, prob in sn_probs.items() if prob >= threshold]
    missed = [obj_id for obj_id, prob in sn_probs.items() if prob < threshold]
    print(f"\nRecovery at threshold={threshold:.4f}: "
          f"{len(recovered)}/{len(known_sn_objects)} "
          f"({len(recovered)/len(known_sn_objects)*100:.0f}%)")

    if missed:
        print(f"  WARNING: Still missing: {missed}")
        print(f"  Consider training longer or with more data.")
    else:
        print(f"  All {len(recovered)} known supernovae RECOVERED!")

    # ==================================================================
    # STEP 4: Apply to entire dataset — write candidates immediately
    # ==================================================================
    print("\n" + "="*60)
    print(f"STEP 4: Applying CNN to all {len(paired)} objects")
    print(f"  Candidates written to {args.output_csv} as they are found")
    print("="*60)

    all_probs_csv = args.output_csv.replace(".csv", "_all_probs.csv")

    start_time = time.time()
    num_candidates = 0
    num_known_found = 0
    completed = 0
    total = len(paired)

    # Open both CSV files and write headers — candidates are flushed immediately
    csv_file = open(args.output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["object_id", "probability", "known_sn"])

    all_probs_file = open(all_probs_csv, "w", newline="")
    all_probs_writer = csv.writer(all_probs_file)
    all_probs_writer.writerow(["object_id", "probability", "known_sn"])

    try:
        for obj_id in sorted(paired.keys()):
            files = paired[obj_id]
            prob = predict_object(
                model, files["hst"], files["jwst1"], files["jwst2"], device)

            is_known = "YES" if obj_id in KNOWN_SUPERNOVAE_SET else "NO"

            # Write ALL probabilities to the full file
            all_probs_writer.writerow([obj_id, f"{prob:.6f}", is_known])

            # Write candidate immediately if above threshold
            if prob >= threshold:
                csv_writer.writerow([obj_id, f"{prob:.4f}", is_known])
                csv_file.flush()  # Flush so it's visible immediately
                num_candidates += 1

                tag = "KNOWN" if is_known == "YES" else "NEW"
                print(f"  *** SN CANDIDATE: {obj_id} (prob={prob:.4f}) [{tag}] ***")

                if is_known == "YES":
                    num_known_found += 1

            completed += 1
            if completed % 5000 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                all_probs_file.flush()
                print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%) "
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

    # Re-read candidates to do final checks
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
    if known_recovered:
        for obj_id in sorted(known_recovered):
            print(f"  FOUND: {obj_id}")
    if known_missed:
        print(f"MISSED known SN:")
        for obj_id in sorted(known_missed):
            print(f"  MISSED: {obj_id}")

    new_count = len(candidate_ids) - len(known_recovered)
    print(f"\nTotal candidates: {len(candidate_ids)} "
          f"(known: {len(known_recovered)}, new: {new_count})")
    print(f"Threshold used: {threshold:.4f}")
    print(f"Detection rate: {detection_rate:.1f} per 1000 objects")

    if detection_rate > 5:
        print(f"WARNING: Detection rate ({detection_rate:.1f}/1000) is higher than "
              f"expected (~1/1000). Consider raising threshold.")

    print(f"\nCandidate file: {args.output_csv} (check anytime, even while running)")
    print(f"All probabilities: {all_probs_csv}")


if __name__ == "__main__":
    main()
