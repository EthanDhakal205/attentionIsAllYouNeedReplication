# Attention Is All You Need

PyTorch replication of the Transformer from Vaswani et al. (2017).
Starting with the model architecture, then a small copy task to check training.

## Setup

Use Python 3.10 or newer.

```sh
python -m venv .venv
```

Activate the virtual environment:
`.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS.

Then install PyTorch:

```sh
python -m pip install -r requirements.txt
```

## Paper

[Attention Is All You Need](papers/attention-is-all-you-need.pdf) by Ashish Vaswani,
Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin.

The included PDF is [arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762v7).
