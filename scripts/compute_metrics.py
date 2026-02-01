import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines()]


def exact_match_rate(references: list[str], predictions: list[str]) -> float:
    if not references:
        return 0.0
    matches = sum(1 for ref, pred in zip(references, predictions) if ref == pred)
    return matches / len(references)


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def compute_code_bleu(references: list[str], predictions: list[str], lang: str) -> Optional[float]:
    if not module_available("tree_sitter"):
        print("CodeBLEU skipped: tree_sitter is not available.")
        return None
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from evaluator.CodeBLEU.code_bleu import calculate_code_bleu_from_lists

    return calculate_code_bleu_from_lists([references], predictions, lang=lang)


def compute_code_bert(references: list[str], predictions: list[str], lang: str) -> Optional[Dict[str, float]]:
    if not module_available("code_bert_score"):
        print("CodeBERT skipped: code_bert_score is not available.")
        return None
    import code_bert_score
    from torch import mean

    precision, recall, f1, f3 = code_bert_score.score(cands=predictions, refs=references, lang=lang)
    precision = mean(precision).item()
    recall = mean(recall).item()
    f1 = mean(f1).item()
    f3 = mean(f3).item()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f3": f3,
    }


def run_bandit(path: str) -> Optional[str]:
    if shutil.which("bandit") is None:
        print("Bandit skipped: bandit is not installed.")
        return None
    if not os.path.exists(path):
        print(f"Bandit skipped: path not found: {path}")
        return None
    result = subprocess.run(["bandit", "-r", path], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        print("Bandit failed:")
        print(result.stderr)
        return None
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CodeBLEU, CodeBERT, exact match, and Bandit metrics.")
    parser.add_argument("--references", required=True, help="Path to reference (gold) file.")
    parser.add_argument("--predictions", required=True, help="Path to prediction file.")
    parser.add_argument("--lang", default="python", help="Language for metrics (default: python).")
    parser.add_argument("--bandit-path", help="Path to run Bandit against (optional).")
    args = parser.parse_args()

    references = read_lines(args.references)
    predictions = read_lines(args.predictions)

    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same number of lines.")

    print(f"Total samples: {len(references)}")

    bleu = compute_code_bleu(references, predictions, args.lang)
    if bleu is not None:
        print(f"CodeBLEU: {bleu}")

    code_bert = compute_code_bert(references, predictions, args.lang)
    if code_bert is not None:
        print(
            "CodeBERT: "
            f"precision={code_bert['precision']}, "
            f"recall={code_bert['recall']}, "
            f"f1={code_bert['f1']}, "
            f"f3={code_bert['f3']}"
        )

    match_rate = exact_match_rate(references, predictions)
    print(f"Exact match rate: {match_rate}")

    if args.bandit_path:
        bandit_output = run_bandit(args.bandit_path)
        if bandit_output:
            print("Bandit output:")
            print(bandit_output)


if __name__ == "__main__":
    main()
