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

Then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

## Tokenizer

The English-German experiment in [section 5.1 of the paper](https://arxiv.org/html/1706.03762v7#S5.SS1)
used BPE with a shared vocabulary of about 37,000 tokens. This implementation
uses [subword-nmt](https://github.com/rsennrich/subword-nmt) to learn character-based
BPE merges jointly from the two languages. English-French used a different,
32,000-token word-piece setup, which is not implemented here.

Prepare UTF-8 training files with one sentence per line and words/punctuation
already separated by spaces, for example `Hello , world !`. Apply the same
word tokenization to new text before encoding it. The tokenizer preserves case
and does not perform normalization or Moses tokenization itself.

```sh
python train_tokenizer.py --source data/train.en --target data/train.de --output tokenizer.json --merges 32000
```

Train on the training split only, then reuse the saved tokenizer for validation,
testing, and inference. The merge count is configurable and is not the vocabulary
size. The final vocabulary depends on the corpus; 32,000 merges is a starting
setting here, not a value specified by the paper. No original trained vocabulary
is bundled, so this reproduces the BPE method rather than the exact WMT token IDs.

```python
from transformer_paper import BPETokenizer, TransformerConfig, make_transformer

tokenizer = BPETokenizer.load("tokenizer.json")
src_ids = tokenizer.encode("Hello , world !")
tgt_ids = tokenizer.encode("Hallo , Welt !", add_bos=True)

config = TransformerConfig.base(len(tokenizer), len(tokenizer), pad_idx=tokenizer.pad_id)
model = make_transformer(config)
```

IDs 0, 1, 2, and 3 are `<pad>`, `<bos>`, `<eos>`, and `<unk>`. These IDs follow this
project's convention. Source sequences end in EOS; target sequences include BOS
and EOS. For training, use `tgt_ids[:-1]` as decoder input and `tgt_ids[1:]` as labels.
Pad batches with `tokenizer.pad_id` and use the existing mask helpers.

`tokenizer.tokenize(text)` returns subword pieces with `@@` continuation markers.
`tokenizer.decode(ids)` joins those pieces, skips BOS/padding, and stops at EOS.
Its output is still word-tokenized text, so punctuation spacing is retained.
New words can fall back to characters seen during training; unseen characters
become `<unk>`. Save the tokenizer with model checkpoints to keep IDs consistent.

## Paper

[Attention Is All You Need](papers/attention-is-all-you-need.pdf) by Ashish Vaswani,
Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin.

The included PDF is [arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762v7).
