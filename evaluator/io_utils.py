from pathlib import Path
from typing import Iterable


def write_lines(path: str, lines: Iterable[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_references_predictions(references_path: str, predictions_path: str,
                                 references: Iterable[str], predictions: Iterable[str]) -> None:
    write_lines(references_path, references)
    write_lines(predictions_path, predictions)
