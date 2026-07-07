"""CNN glyph classifier — a learned fallback for worn/ambiguous characters.
Loads glyph_cnn.pt (from train_classifier.py). If torch or the model file is
missing, classify() returns None and the caller keeps template matching."""
import cv2
import numpy as np

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?" 
MODEL_PATH = "glyph_cnn.pt"
_net = None
_failed = False
SIZE = 40

def _build_net():
    import torch.nn as nn
    import torch.nn.functional as Fn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 32, 3, padding=1)
            self.c3 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 37)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            x = Fn.max_pool2d(Fn.relu(self.c1(x)), 2)
            x = Fn.max_pool2d(Fn.relu(self.c2(x)), 2)
            x = Fn.max_pool2d(Fn.relu(self.c3(x)), 2)
            x = x.flatten(1)
            x = Fn.relu(self.fc1(x))
            x = self.drop(x)
            return self.fc2(x)
    return Net()


def _load():
    global _net, _failed
    if _net is None and not _failed:
        try:
            import torch
            _net = _build_net()
            _net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            _net.eval()
        except Exception:
            _failed = True   # no torch or no model -> silently disable
    return _net


def _prep(ink):
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return np.zeros((SIZE, SIZE), np.float32)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    sc = 28.0 / max(h, w)
    nh, nw = max(1, int(h * sc)), max(1, int(w * sc))
    r = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas.astype(np.float32) / 255.0


def classify(q):
    """q is a height-normalised glyph, ink=black(0) on white(255) (the _norm output).
    Returns (char, confidence) or None if the model isn't available."""
    net = _load()
    if net is None:
        return None
    import torch
    import torch.nn.functional as Fn
    x = torch.tensor(_prep(255 - q))[None, None]
    with torch.no_grad():
        p = Fn.softmax(net(x), 1)[0]
    i = int(p.argmax())
    if CHARS[i] == "?":
        return None          # model says "not a character" -> caller drops it
    return CHARS[i], float(p[i])