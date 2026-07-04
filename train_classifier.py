import cv2, numpy as np, random, time
from PIL import Image, ImageDraw, ImageFont
import torch, torch.nn as nn, torch.nn.functional as Fn
from emb_plate_reader import ensure_font   # reuse the same font loader

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
IDX = {c: i for i, c in enumerate(CHARS)}
SIZE = 40
rng = random.Random(0); np.random.seed(0); torch.manual_seed(0)

font = ImageFont.truetype(ensure_font(), 200)


def render_ink(ch):
    im = Image.new("L", (320, 340), 255); d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), ch, font=font); w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((320 - w) / 2 - bb[0], (340 - h) / 2 - bb[1]), ch, fill=0, font=font)
    a = cv2.threshold(np.array(im), 128, 255, cv2.THRESH_BINARY)[1]
    ink = 255 - a
    ys, xs = np.where(ink > 0)
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def prep(ink):
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


def augment(ink):
    a = ink.copy()
    ang = rng.uniform(-10, 10)
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    k = rng.choice([0, 2, 3, 3, 4])
    if k:
        ker = np.ones((k, k), np.uint8)
        a = cv2.erode(a, ker) if rng.random() < 0.6 else cv2.dilate(a, ker)
    if rng.random() < 0.5:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.6, 2.0))
        a = (a > 90).astype(np.uint8) * 255
    if rng.random() < 0.5:                      # random edge clip (simulates cropped glyphs)
        hh, ww = a.shape; frac = rng.uniform(0.05, 0.20)
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "bottom": a[int(hh * (1 - frac)):, :] = 0
        elif side == "top":  a[:int(hh * frac), :] = 0
        elif side == "left": a[:, :int(ww * frac)] = 0
        else:                a[:, int(ww * (1 - frac)):] = 0
    if rng.random() < 0.4:                       # speckle noise
        noise = (np.random.rand(*a.shape) < 0.03).astype(np.uint8) * 255
        a = cv2.bitwise_or(a, noise); a = cv2.medianBlur(a, 3)
    return a


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 36); self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = Fn.max_pool2d(Fn.relu(self.c1(x)), 2)
        x = Fn.max_pool2d(Fn.relu(self.c2(x)), 2)
        x = Fn.max_pool2d(Fn.relu(self.c3(x)), 2)
        x = x.flatten(1); x = Fn.relu(self.fc1(x)); x = self.drop(x)
        return self.fc2(x)


def main():
    t0 = time.time()
    bases = {c: render_ink(c) for c in CHARS}
    PER = 450
    X, Y = [], []
    for c in CHARS:
        for _ in range(PER):
            X.append(prep(augment(bases[c]))); Y.append(IDX[c])
    X = np.stack(X)[:, None]; Y = np.array(Y)
    perm = np.random.permutation(len(X)); X, Y = X[perm], Y[perm]
    ntr = int(0.9 * len(X))
    Xtr, Ytr = torch.tensor(X[:ntr]), torch.tensor(Y[:ntr])
    Xva, Yva = torch.tensor(X[ntr:]), torch.tensor(Y[ntr:])
    print(f"dataset {X.shape} built in {time.time() - t0:.1f}s")

    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(20):
        net.train(); idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]; opt.zero_grad()
            Fn.cross_entropy(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                acc = (net(Xva).argmax(1) == Yva).float().mean().item()
            print(f"epoch {ep + 1:2d}  val_acc {acc:.3f}")
    torch.save(net.state_dict(), "glyph_cnn.pt")
    print("saved glyph_cnn.pt")


if __name__ == "__main__":
    main()