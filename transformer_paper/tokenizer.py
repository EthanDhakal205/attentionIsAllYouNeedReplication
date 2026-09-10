import io
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe


class BPETokenizer:
    """Joint source/target BPE for text that has already been word-tokenized."""

    special_tokens = ("<pad>", "<bos>", "<eos>", "<unk>")
    pad_id, bos_id, eos_id, unk_id = range(4)

    def __init__(self, codes: str, vocabulary: Sequence[str]) -> None:
        self.vocabulary = list(vocabulary)
        if self.vocabulary[:4] != list(self.special_tokens):
            raise ValueError("vocabulary must start with <pad>, <bos>, <eos>, <unk>")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary contains duplicate tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.codes = codes
        # subword-nmt needs an explicit zero for a codes file with no merges.
        merges = -1 if len(codes.strip().splitlines()) > 1 else 0
        self.bpe = BPE(io.StringIO(codes), merges=merges, vocab=set(self.vocabulary))

    def __len__(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(
        cls,
        files: Sequence[str | Path],
        num_merges: int = 32000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn shared merges and IDs from both languages' training files."""
        if num_merges < 0 or min_frequency < 1:
            raise ValueError("num_merges must be nonnegative and min_frequency must be positive")
        words = Counter()
        for path in files:
            with Path(path).open(encoding="utf-8") as source:
                for line in source:
                    words.update(line.split())
        if not words:
            raise ValueError("training files contain no tokens")
        if any(token in words for token in cls.special_tokens):
            raise ValueError("training text contains a reserved special token")

        codes = io.StringIO()
        if num_merges and any(len(word) > 1 for word in words):
            counts = io.StringIO("".join(f"{word} {count}\n" for word, count in sorted(words.items())))
            learn_bpe(counts, codes, num_merges, min_frequency=min_frequency, is_dict=True)
        else:
            codes.write("#version: 0.2\n")

        code_text = codes.getvalue()
        merges = -1 if len(code_text.strip().splitlines()) > 1 else 0
        bpe = BPE(io.StringIO(code_text), merges=merges)
        counts = Counter()
        for word, frequency in words.items():
            for piece in bpe.segment_tokens([word]):
                counts[piece] += frequency

        # Keep character fallbacks for new words made from the training alphabet.
        for char in set("".join(words)):
            counts.setdefault(char, 0)
            counts.setdefault(char + "@@", 0)
        vocabulary = list(cls.special_tokens)
        vocabulary.extend(
            sorted((piece for piece in counts if piece not in cls.special_tokens),
                   key=lambda piece: (-counts[piece], piece))
        )
        return cls(code_text, vocabulary)

    def tokenize(self, text: str) -> list[str]:
        return self.bpe.segment_tokens(text.split())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(piece, self.unk_id) for piece in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Remove BPE boundaries, returning word-tokenized text."""
        pieces = []
        for index in ids:
            if not 0 <= index < len(self):
                raise ValueError(f"token ID out of range: {index}")
            if index == self.eos_id:
                break
            if index not in (self.pad_id, self.bos_id):
                pieces.append(self.vocabulary[index])
        text = " ".join(pieces).replace("@@ ", "")
        return text.removesuffix("@@")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"codes": self.codes, "vocabulary": self.vocabulary}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["codes"], data["vocabulary"])
