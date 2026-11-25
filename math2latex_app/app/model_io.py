"""Model I/O: loads the trained bundle and reproduces the notebook's inference exactly.

The classes and preprocessing here MUST match the training notebook token-for-token —
same architecture, same canvas size, same normalization — or predictions degrade silently.
"""
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ----------------------------------------------------------------- architecture
class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1),
                      nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers
        self.cnn = nn.Sequential(
            *block(1, 64, True), *block(64, 128, True),
            *block(128, 256, True), *block(256, d_model, False),
        )
        self.pos2d = None

    def _build_pos2d(self, h, w, d, device):
        def sincos(n, d):
            pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
            pe = torch.zeros(n, d)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe
        row = sincos(h, d // 2).unsqueeze(1).expand(h, w, d // 2)
        col = sincos(w, d // 2).unsqueeze(0).expand(h, w, d // 2)
        return torch.cat([row, col], dim=-1).reshape(h * w, d).to(device)

    def forward(self, x):
        f = self.cnn(x)
        B, C, H, W = f.shape
        f = f.flatten(2).transpose(1, 2)
        if self.pos2d is None or self.pos2d.shape[0] != H * W:
            self.pos2d = self._build_pos2d(H, W, C, f.device)
        return f + self.pos2d


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Math2LaTeX(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, ff=1024, dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos1d = PositionalEncoding1D(d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)


# ----------------------------------------------------------------- runtime
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class Predictor:
    def __init__(self, bundle_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bundle = torch.load(bundle_path, map_location=self.device, weights_only=False)
        cfg = bundle["config"]
        self.itos = bundle["itos"]
        self.img_h, self.img_w = cfg["img_h"], cfg["img_w"]
        self.max_len = cfg["max_len"]
        self.model = Math2LaTeX(len(self.itos), d_model=cfg["d_model"]).to(self.device)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

    def autocrop(self, img: Image.Image, margin_frac: float = 0.06) -> Image.Image:
        """Trim to the bounding box of the ink, with a small proportional margin.

        WHY THIS MATTERS: the app's canvas is large (1200x520), but a written
        equation may occupy only a small corner of it. Without cropping, the
        whole canvas gets scaled down to fit 96x480 and the strokes end up a
        dozen pixels tall — nothing like the training data, where MathWriting
        inks were rendered tightly framed. Cropping first makes the model's
        input match the distribution it was trained on.
        """
        g = img.convert("L")
        a = np.array(g, dtype=np.float32)
        if a.mean() > 127:            # dark ink on light paper -> invert so ink is high
            a = 255.0 - a
        ink = a > 40                  # threshold away antialiasing and paper texture
        if not ink.any():
            return img               # blank canvas: nothing to crop
        rows = np.where(ink.any(axis=1))[0]
        cols = np.where(ink.any(axis=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        # margin scaled to the ink size, so small and large writing crop alike
        m = max(4, int(round(margin_frac * max(y1 - y0, x1 - x0))))
        y0, y1 = max(0, y0 - m), min(a.shape[0], y1 + m)
        x0, x1 = max(0, x0 - m), min(a.shape[1], x1 + m)
        return img.crop((x0, y0, x1, y1))

    # identical to the notebook, but fed an autocropped image
    def preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        a = np.array(img, dtype=np.float32) / 255.0
        if a.mean() > 0.5:
            a = 1.0 - a
        img = Image.fromarray((a * 255).astype(np.uint8))
        w, h = img.size
        scale = min(self.img_h / h, self.img_w / w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("L", (self.img_w, self.img_h), 0)
        canvas.paste(img, (0, (self.img_h - nh) // 2))
        x = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0)
        return ((x - 0.5) / 0.5).unsqueeze(0)

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, SOS_ID):
                continue
            out.append(self.itos[i])
        s = ""
        for t in out:
            s += t + " " if (t.startswith("\\") and len(t) > 1 and t[1].isalpha()) else t
        return re.sub(r"\s+", " ", s).strip()

    @torch.no_grad()
    def predict(self, img: Image.Image, beam: int = 1) -> str:
        """Image -> LaTeX.

        beam=1 is greedy decoding: at each step take the single most likely next
        token. Fast, but a token that looks best in isolation can lead into a
        sequence the model would have scored lower overall — and there's no way
        to back out.

        beam>1 keeps the `beam` best partial sequences alive at every step and
        picks the best complete one at the end, so an early token that only
        looks good locally can be overtaken by an alternative that pays off
        later. Costs roughly `beam` times the compute; 4-5 is the usual sweet spot.
        """
        img = self.autocrop(img)                 # crop to the ink before anything else
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        memory = self.model.encoder(x)
        return (self._greedy(memory) if beam <= 1 else self._beam(memory, beam))

    def _step_logprobs(self, seq, memory):
        """Log-probabilities over the vocabulary for the next token after `seq`."""
        t = self.model.pos1d(self.model.embed(seq))
        L = seq.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=self.device), diagonal=1)
        h = self.model.decoder(t, memory.expand(seq.size(0), -1, -1), tgt_mask=causal)
        return torch.log_softmax(self.model.head(h[:, -1]), dim=-1)

    def _greedy(self, memory) -> str:
        seq = torch.tensor([[SOS_ID]], device=self.device)
        for _ in range(self.max_len):
            next_id = self._step_logprobs(seq, memory).argmax(-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
            if next_id.item() == EOS_ID:
                break
        return self.decode(seq[0].tolist())

    def _beam(self, memory, beam: int, length_penalty: float = 0.7) -> str:
        # live beams share one tensor; `scores` holds cumulative log-probability
        seqs = torch.tensor([[SOS_ID]], device=self.device)
        scores = torch.zeros(1, device=self.device)
        finished = []                                    # (normalized_score, token_ids)

        for _ in range(self.max_len):
            logp = self._step_logprobs(seqs, memory)      # (live, vocab)
            total = scores.unsqueeze(1) + logp            # extend every live beam
            flat = total.view(-1)
            k = min(beam, flat.numel())
            top, idx = flat.topk(k)
            beam_idx = torch.div(idx, logp.size(1), rounding_mode="floor")
            tok_idx = idx % logp.size(1)
            seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores = top

            # retire any beam that just emitted <eos>
            done = tok_idx == EOS_ID
            if done.any():
                for i in torch.nonzero(done).flatten().tolist():
                    # normalize by length so short sequences aren't unfairly favored
                    norm = scores[i].item() / (seqs.size(1) ** length_penalty)
                    finished.append((norm, seqs[i].tolist()))
                keep = ~done
                if not keep.any():
                    break
                seqs, scores = seqs[keep], scores[keep]
            if len(finished) >= beam:
                break

        if not finished:                                  # hit the length cap
            norm = scores[0].item() / (seqs.size(1) ** length_penalty)
            finished.append((norm, seqs[0].tolist()))
        return self.decode(max(finished, key=lambda p: p[0])[1])
