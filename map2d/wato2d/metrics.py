"""Streaming confusion matrix -> per-class IoU. No dependencies beyond numpy."""
import numpy as np

from .classes import CLASSES, IGNORE


class Confusion:
    def __init__(self, n=len(CLASSES)):
        self.m = np.zeros((n, n), np.int64)

    def update(self, pred, gt):
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)
        ok = gt != IGNORE
        idx = gt[ok] * self.m.shape[0] + pred[ok]
        self.m += np.bincount(idx, minlength=self.m.size).reshape(self.m.shape)

    def iou(self):
        out = {}
        for c, nm in enumerate(CLASSES):
            tp = self.m[c, c]
            fp = self.m[:, c].sum() - tp
            fn = self.m[c, :].sum() - tp
            present = int(self.m[c, :].sum())
            out[nm] = float(tp / present) if present else float("nan")
        # average only over classes that actually occur: a class with no GT
        # pixels is undefined, not zero (this is the mistake that made the
        # class-parity numbers wrong earlier)
        vals = [out[n] for n in CLASSES[1:] if not np.isnan(out[n])]
        out["miou"] = float(np.mean(vals)) if vals else 0.0
        return out
