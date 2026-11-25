# Math2LaTeX

Write a math expression by hand, get LaTeX back.

This is an end-to-end handwriting recognition system built from scratch: a CNN
encoder paired with a Transformer decoder, trained in PyTorch on Google's
[MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting)
dataset (~230,000 handwritten expressions), and served as a web app where you draw
on a canvas and get typeset math back.

It was built as a first deep learning project, so the repository is written to be
read: [`math2latex.ipynb`](math2latex.ipynb) walks through the whole pipeline —
parsing pen strokes, building a LaTeX tokenizer, designing the architecture,
training, and evaluating, with each section explained for someone who hasn't done
this before.

## Results

Measured on a held-out test set of unseen writers, against
[pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) as a baseline:

| Model | Exact match | Token error rate |
|---|---|---|
| This model (trained on MathWriting) | 0.638 | 0.078 |
| pix2tex (off-the-shelf) | 0.000 | 4.614 |
(as of November 2026)

Both models are scored by the same code, with LaTeX-aware normalization so neither
is penalized for notation conventions (`\dfrac` vs `\frac`, `x^{2}` vs `x^2`).
Note: this model is trained and tested on MathWriting, while pix2tex
is evaluated out of domain.

## How it works

The image is read by a convolutional encoder and turned into a sequence of 720
feature vectors, which a 4-layer Transformer decoder attends over while generating
LaTeX one token at a time — the same architecture used for image captioning, with
LaTeX in place of a caption. Two inference-time additions improve accuracy without
retraining: the input is auto-cropped to the bounding box of the ink, and decoding
uses beam search with a length penalty rather than greedy argmax.
