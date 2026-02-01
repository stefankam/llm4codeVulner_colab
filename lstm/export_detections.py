import argparse
from pathlib import Path


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LSTM-detected vulnerable snippets for LLM repair."
    )
    parser.add_argument(
        "--examples-dir",
        default="lstm_examples",
        help="Directory containing example vulnerable files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/lstm_detections",
        help="Output directory for prompts.txt (and optional references.txt).",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Optional directory containing fixed versions matched by filename.",
    )
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    references = []
    for file_path in sorted(examples_dir.glob("*.py")):
        prompts.append(read_file(file_path))
        if args.labels_dir:
            label_path = Path(args.labels_dir) / file_path.name
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file: {label_path}")
            references.append(read_file(label_path))

    (output_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    if args.labels_dir:
        (output_dir / "references.txt").write_text("\n".join(references) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
