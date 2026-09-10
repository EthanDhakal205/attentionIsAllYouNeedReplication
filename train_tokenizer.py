import argparse

from transformer_paper import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shared BPE on word-tokenized parallel text.")
    parser.add_argument("--source", required=True, help="Source-language training file (UTF-8)")
    parser.add_argument("--target", required=True, help="Target-language training file (UTF-8)")
    parser.add_argument("--output", required=True, help="Where to save the tokenizer JSON")
    parser.add_argument("--merges", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    tokenizer = BPETokenizer.train(
        [args.source, args.target], num_merges=args.merges, min_frequency=args.min_frequency,
    )
    tokenizer.save(args.output)
    print(f"Saved {len(tokenizer)} tokens to {args.output}")


if __name__ == "__main__":
    main()
