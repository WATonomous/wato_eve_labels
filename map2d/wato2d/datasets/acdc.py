"""ACDC (Adverse Conditions Dataset with Correspondences) -> our 4 classes.

2,006 labelled images: 400 fog / 400 night / 400 rain / 400 snow in train, plus
100/106/100/100 in val. This is the only adverse-weather corpus with fine pixel
labels, which is why it matters for the night target.

Layout (from the dataset README):
  {root}/rgb_anon/{condition}/{split}/{seq}/{seq}_frame_{frame:06d}_rgb_anon.png
  {root}/gt/{condition}/{split}/{seq}/{seq}_frame_{frame:06d}_gt_labelIds.png
  {root}/gt/{condition}/{split}/{seq}/{seq}_frame_{frame:06d}_gt_invIds.png

`labelIds` are **Cityscapes ids**, not trainIds -- road is 7, sidewalk is 8.
`invIds` is the invalid mask (1 = do not train here); there is no separate masks
download, it ships inside gt_trainval.zip.
"""
import pathlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..classes import CLASSES, IGNORE

# Cityscapes labelIds -> ours. 'ground' and 'parking' are drivable; 'terrain'
# (grass/soil) is not, so it stays `other`.
CS_TO_OURS = {
    6: "drivable",    # ground
    7: "drivable",    # road
    9: "drivable",    # parking
    8: "walkway",     # sidewalk
}
# 0 unlabeled, 1 ego vehicle, 2 rectification border, 3 out of roi
CS_IGNORE = {0, 1, 2, 3}

CONDITIONS = ["fog", "night", "rain", "snow"]


class ACDC(Dataset):
    def __init__(self, root, split="train", size=512, augment=True, conditions=None):
        self.root = pathlib.Path(root)
        self.size = size
        self.augment = augment
        self.items = []
        missing = 0
        for cond in (conditions or CONDITIONS):
            gdir = self.root / "gt" / cond / split
            if not gdir.exists():
                continue
            for lab in sorted(gdir.glob("*/*_gt_labelIds.png")):
                seq = lab.parent.name
                stem = lab.name.replace("_gt_labelIds.png", "")
                img = self.root / "rgb_anon" / cond / split / seq / f"{stem}_rgb_anon.png"
                inv = lab.with_name(lab.name.replace("_gt_labelIds.png", "_gt_invIds.png"))
                # RGB and GT come from separate downloads; a partial RGB archive
                # (we recovered 3 of 4 train conditions) must not crash training
                if not img.exists():
                    missing += 1
                    continue
                self.items.append((img, lab, inv, cond))
            if missing:
                print(f"  acdc {cond}/{split}: skipped {missing} GT entries with no RGB")
        if not self.items:
            raise RuntimeError(f"no ACDC samples under {self.root}")
        self.lut = np.full(256, CLASSES.index("other"), np.uint8)
        self.lut[list(CS_IGNORE)] = IGNORE
        for k, name in CS_TO_OURS.items():
            self.lut[k] = CLASSES.index(name)
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_p, lab_p, inv_p, _ = self.items[i]
        img = Image.open(img_p).convert("RGB")
        lab = np.array(Image.open(lab_p), np.uint8)
        lab = self.lut[np.clip(lab, 0, 255)]
        if inv_p.exists():
            inv = np.array(Image.open(inv_p), np.uint8)
            lab[inv > 0] = IGNORE          # unlabelable region (e.g. glare, fog)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = np.ascontiguousarray(lab[:, ::-1])
        lab = np.array(Image.fromarray(lab).resize((self.size, self.size), Image.NEAREST))
        return self.tf(img), torch.from_numpy(lab.astype(np.int64))


class Concat(torch.utils.data.Dataset):
    """ADE20K + ACDC, sampling equally so the clean data does not swamp the
    adverse data."""

    def __init__(self, a, b, ratio=0.5):
        self.a, self.b, self.ratio = a, b, ratio

    def __len__(self):
        return int(len(self.b) / self.ratio) if self.ratio else len(self.a)

    def __getitem__(self, i):
        if random.random() < self.ratio:
            return self.b[random.randrange(len(self.b))]
        return self.a[random.randrange(len(self.a))]
