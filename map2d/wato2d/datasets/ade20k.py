"""ADE20K scene parsing -> our 4-class driving ontology.

ADE20K is the one large segmentation corpus that downloads without a login, so
it is the autonomous pretraining set. Only a handful of its 150 classes matter
here: road/sidewalk/earth map onto drivable/walkway/crossing, everything else is
`other`. That is enough to teach "what a road looks like" before we fine-tune on
the car, and it is deliberately conservative.

  dataset = ADE20K(root, split="training", size=512)
"""
import pathlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..classes import CLASSES, IGNORE

# ADE20K class name -> our class. Names come from objectInfo150.txt.
NAME_MAP = {
    "road": "drivable",
    "sidewalk": "walkway",
    "earth": "drivable",
    "grass": "walkway",      # verges read like walkway at BEV resolution
}
# also catch a few names that vary between ADE20K releases
NAME_MAP.update({"runway": "drivable", "path": "walkway", "floor": "other"})


def read_class_names(root):
    """objectInfo150.txt: tab-separated `Idx Ratio Train Val Name`.

    Names carry synonyms ("road, route"), so keep the first one. Rows are
    1-based and the annotation PNGs use the same 1-based index.
    """
    txt = pathlib.Path(root) / "objectInfo150.txt"
    names = []
    for line in txt.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        names.append(parts[4].split(",")[0].strip())
    return names


class ADE20K(Dataset):
    def __init__(self, root, split="training", size=512, augment=True):
        self.root = pathlib.Path(root)
        self.img_dir = self.root / "images" / split
        self.ann_dir = self.root / "annotations" / split
        self.files = sorted(p.name[:-4] for p in self.img_dir.glob("*.jpg"))
        if not self.files:
            raise RuntimeError(f"no images under {self.img_dir}")
        names = read_class_names(self.root)
        # every ADE20K pixel is labelled (no void), so unmapped classes become
        # `other` -- the model needs negatives or it learns "everything is road"
        self.lut = np.full(256, CLASSES.index("other"), np.uint8)
        self.lut[0] = IGNORE                      # 0 does not occur; keep it safe
        for i, nm in enumerate(names, start=1):
            if nm in NAME_MAP:
                self.lut[i] = CLASSES.index(NAME_MAP[nm])
        self.size = size
        self.augment = augment
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        img = Image.open(self.img_dir / f"{n}.jpg").convert("RGB")
        ann = Image.open(self.ann_dir / f"{n}.png")
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ann = ann.transpose(Image.FLIP_LEFT_RIGHT)
        lab = np.array(ann.resize((self.size, self.size), Image.NEAREST), np.uint8)
        lab = self.lut[lab]
        # ADE20K has no crossing class; do not pretend it does (supervised later
        # on the car data instead)
        lab[lab == CLASSES.index("crossing")] = IGNORE
        return self.tf(img), torch.from_numpy(lab.astype(np.int64))


def check_stats(root):
    """Print how much of each class the mapping actually produces."""
    ds = ADE20K(root, "training", 256, augment=False)
    hist = np.zeros(len(CLASSES) + 1, np.int64)
    for i in range(0, len(ds), max(1, len(ds) // 40)):
        _, y = ds[i]
        v = y.numpy()
        for c in range(len(CLASSES)):
            hist[c] += int((v == c).sum())
        hist[-1] += int((v == IGNORE).sum())
    tot = hist.sum()
    for c, nm in enumerate(CLASSES):
        print(f"  {nm:9s} {100 * hist[c] / tot:6.2f}%")
    print(f"  {'ignore':9s} {100 * hist[-1] / tot:6.2f}%")
