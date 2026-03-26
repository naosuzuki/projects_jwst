"""
Train a CNN to Detect Supernovae by Comparing JWST and HST PNG Images

Pipeline:
1. Load known supernovae (23 confirmed) as positive training examples
2. Generate artificial supernovae by injecting point sources into images
3. Sample negative examples (objects with no supernova)
4. Train a CNN on (HST, JWST) image pairs
5. Validate recovery of the 23 known supernovae
6. Apply to all matched objects and save candidates

Usage:
    python train_supernova_cnn.py [options]
    python train_supernova_cnn.py --epochs 50 --num-workers 8
"""

import argparse
import csv
import glob
import os
import re
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

IMAGE_SIZE = 64  # Resize all images to 64x64
NUM_CHANNELS = 3  # HST, JWST1, JWST2


# --- Data loading and file matching ---

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


# --- Artificial supernova generation ---

def add_artificial_supernova(image, num_sn=1):
    """Inject artificial point sources (supernovae) into an image.

    Adds Gaussian-shaped bright sources at random positions.
    Returns modified image and list of (x, y) injection positions.
    """
    img = image.copy()
    h, w = img.shape
    positions = []

    for _ in range(num_sn):
        # Random position (avoid edges)
        x = random.randint(5, w - 6)
        y = random.randint(5, h - 6)

        # Random brightness and size
        peak = random.uniform(0.3, 0.9)
        sigma = random.uniform(1.0, 3.0)

        # Create Gaussian PSF
        yy, xx = np.mgrid[0:h, 0:w]
        gaussian = peak * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        img = np.clip(img + gaussian, 0, 1)
        positions.append((x, y))

    return img, positions


# --- Dataset ---

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
    """Dataset of (HST, JWST1, JWST2) image triplets with supernova labels.

    Each sample is a 3-channel image: channel 0 = HST, channel 1 = JWST1, channel 2 = JWST2.
    Label: 1 = supernova present, 0 = no supernova.
    """

    def __init__(self, samples, augment=False):
        """
        Args:
            samples: list of (hst_path, jwst1_path, jwst2_path, label) tuples
            augment: apply random augmentation
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hst_path, jwst1_path, jwst2_path, label = self.samples[idx]

        hst_img = load_and_resize(hst_path)
        jwst1_img = load_and_resize(jwst1_path)
        jwst2_img = load_and_resize(jwst2_path)

        if hst_img is None or jwst1_img is None or jwst2_img is None:
            return torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(0.0)

        if self.augment:
            hst_img, jwst1_img, jwst2_img = augment_images([hst_img, jwst1_img, jwst2_img])

        triplet = np.stack([hst_img, jwst1_img, jwst2_img], axis=0)
        return torch.tensor(triplet, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class ArtificialSupernovaDataset(Dataset):
    """Generate training data with artificial supernovae injected.

    Physical constraints:
    - JWST supernova: SN appears in BOTH jwst1 and jwst2 (same position),
      but NOT in HST (SN happened after HST observation).
    - HST supernova: SN appears in HST only, NOT in jwst1 or jwst2
      (SN was active during HST epoch, faded by JWST epoch).
    """

    def __init__(self, object_files, num_positive, num_negative, augment=True):
        """
        Args:
            object_files: list of {"hst": path, "jwst1": path, "jwst2": path} dicts
            num_positive: number of artificial SN samples to generate
            num_negative: number of negative (no SN) samples
            augment: apply random augmentation
        """
        self.object_files = object_files
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.total = num_positive + num_negative
        self.augment = augment

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        is_positive = idx < self.num_positive

        # Pick a random object
        files = random.choice(self.object_files)
        hst_img = load_and_resize(files["hst"])
        jwst1_img = load_and_resize(files["jwst1"])
        jwst2_img = load_and_resize(files["jwst2"])

        if hst_img is None or jwst1_img is None or jwst2_img is None:
            return torch.zeros(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(0.0)

        if is_positive:
            if random.random() > 0.5:
                # JWST supernova: inject into BOTH jwst1 and jwst2 at same position
                h, w = jwst1_img.shape
                x = random.randint(5, w - 6)
                y = random.randint(5, h - 6)
                peak = random.uniform(0.3, 0.9)
                sigma = random.uniform(1.0, 3.0)
                yy, xx = np.mgrid[0:h, 0:w]
                gaussian = peak * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                jwst1_img = np.clip(jwst1_img + gaussian, 0, 1).astype(np.float32)
                jwst2_img = np.clip(jwst2_img + gaussian, 0, 1).astype(np.float32)
            else:
                # HST supernova: inject into HST only
                hst_img, _ = add_artificial_supernova(hst_img)
            label = 1.0
        else:
            label = 0.0

        if self.augment:
            hst_img, jwst1_img, jwst2_img = augment_images([hst_img, jwst1_img, jwst2_img])

        triplet = np.stack([hst_img, jwst1_img, jwst2_img], axis=0)
        return torch.tensor(triplet, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# --- CNN Model ---

class SupernovaCNN(nn.Module):
    """CNN for binary classification of (HST, JWST1, JWST2) image triplets.

    Input: 3-channel 64x64 image (HST + JWST1 + JWST2).
    Output: single probability (supernova present or not).
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


# --- Training ---

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


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


# --- Inference ---

def predict_object(model, hst_path, jwst1_path, jwst2_path, device):
    """Predict supernova probability for a single (HST, JWST1, JWST2) triplet."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN to detect supernovae in JWST/HST image pairs."
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
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--num-artificial", type=int, default=5000,
                        help="Number of artificial SN training samples (default: 5000)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--model-path", default="supernova_cnn.pth",
                        help="Path to save trained model (default: supernova_cnn.pth)")
    parser.add_argument("--output-csv", default="supernova_cnn_candidates.csv",
                        help="Output CSV file (default: supernova_cnn_candidates.csv)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build file map — require all 3 images (HST + JWST1 + JWST2)
    file_map = build_file_maps(args.hst_dir, args.jwst_dir)
    paired = {k: v for k, v in file_map.items()
              if "hst" in v and "jwst1" in v and "jwst2" in v}
    print(f"Matched {len(paired)} objects with HST + JWST1 + JWST2")

    # Separate known supernovae and non-supernovae
    known_sn_objects = {k: v for k, v in paired.items() if k in KNOWN_SUPERNOVAE_SET}
    non_sn_objects = {k: v for k, v in paired.items() if k not in KNOWN_SUPERNOVAE_SET}

    print(f"Known supernovae in dataset: {len(known_sn_objects)}/{len(KNOWN_SUPERNOVAE_SET)}")
    if known_sn_objects:
        print(f"  Found IDs: {sorted(known_sn_objects.keys())}")
    missing = KNOWN_SUPERNOVAE_SET - set(known_sn_objects.keys())
    if missing:
        print(f"  Missing IDs: {sorted(missing)}")

    # Expected SN rate: ~1-2 per 1000 objects
    expected_rate = len(KNOWN_SUPERNOVAE_SET) / len(paired) if len(paired) > 0 else 0.001
    print(f"Expected SN rate: ~{expected_rate*1000:.1f} per 1000 objects")

    # --- Build training data ---
    print("\n--- Building training data ---")

    # Positive samples: known supernovae (real) as (hst, jwst1, jwst2) triplets
    real_positive = []
    for obj_id, files in known_sn_objects.items():
        real_positive.append((files["hst"], files["jwst1"], files["jwst2"], 1.0))
    print(f"Real positive samples (known SN): {len(real_positive)}")

    # Collect non-SN objects for artificial generation (need all 3 images)
    non_sn_keys = list(non_sn_objects.keys())
    random.shuffle(non_sn_keys)
    non_sn_file_list = []
    for obj_id in non_sn_keys[:10000]:
        files = non_sn_objects[obj_id]
        non_sn_file_list.append(files)

    # Use highly imbalanced training: many more negatives than positives
    # to match the real ~1/1000 SN rate and reduce false positives
    num_neg = args.num_artificial * 10  # 10:1 negative to positive ratio
    print(f"Generating {args.num_artificial} artificial SN + {num_neg} negative samples")
    art_dataset = ArtificialSupernovaDataset(
        non_sn_file_list,
        num_positive=args.num_artificial,
        num_negative=num_neg,
        augment=True,
    )

    # Also add real known SN as positive samples (oversampled)
    real_sn_samples = real_positive * 20  # Oversample real SN to boost signal
    real_sn_dataset = SupernovaDataset(real_sn_samples, augment=True)

    # Combine datasets
    train_dataset = torch.utils.data.ConcatDataset([art_dataset, real_sn_dataset])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Validation set: real known SN (no augmentation) for checking recovery
    val_dataset = SupernovaDataset(real_positive, augment=False)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Train ---
    print(f"\n--- Training CNN for {args.epochs} epochs ---")
    model = SupernovaCNN().to(device)
    # Use pos_weight to account for class imbalance
    pos_weight = torch.tensor([num_neg / args.num_artificial]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {args.model_path}")

    # Reload best model
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # --- Validate: check recovery of 23 known supernovae ---
    print("\n--- Validating against known supernovae ---")
    recovered = []
    missed = []

    for obj_id, files in known_sn_objects.items():
        prob = predict_object(model, files["hst"], files["jwst1"], files["jwst2"], device)

        if prob >= args.threshold:
            recovered.append((obj_id, prob))
            print(f"  RECOVERED: {obj_id} (prob={prob:.3f})")
        else:
            missed.append((obj_id, prob))
            print(f"  MISSED:    {obj_id} (prob={prob:.3f})")

    print(f"\nRecovery: {len(recovered)}/{len(known_sn_objects)} "
          f"({len(recovered)/max(len(known_sn_objects),1)*100:.1f}%)")

    # --- Apply to all objects ---
    print(f"\n--- Applying CNN to all {len(paired)} objects ---")
    start_time = time.time()
    all_candidates = []
    completed = 0
    total = len(paired)

    for obj_id in sorted(paired.keys()):
        files = paired[obj_id]
        prob = predict_object(model, files["hst"], files["jwst1"], files["jwst2"], device)

        if prob >= args.threshold:
            is_known = "YES" if obj_id in KNOWN_SUPERNOVAE_SET else "NO"
            all_candidates.append([obj_id, f"{prob:.4f}", is_known])

        completed += 1
        if completed % 5000 == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                  f"| {rate:.0f} obj/s | ETA: {eta:.0f}s "
                  f"| Candidates: {len(all_candidates)}")

    elapsed = time.time() - start_time
    print(f"\nInference complete in {elapsed:.1f}s")

    # Save results
    if all_candidates:
        all_candidates.sort(key=lambda r: float(r[1]), reverse=True)

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["object_id", "probability", "known_sn"])
            writer.writerows(all_candidates)
        print(f"Candidates saved to {args.output_csv}")

        known_found = sum(1 for r in all_candidates if r[2] == "YES")
        new_found = sum(1 for r in all_candidates if r[2] == "NO")
        print(f"Total candidates: {len(all_candidates)}")
        print(f"  Known supernovae recovered: {known_found}")
        print(f"  New supernova candidates: {new_found}")
        print(f"  Detection rate: {len(all_candidates)/total*1000:.1f} per 1000 objects")
    else:
        print("\nNo supernova candidates found.")


if __name__ == "__main__":
    main()
