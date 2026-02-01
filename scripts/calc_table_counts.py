import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LlmDatasetStats:
    repos: int
    commits: int
    files: int
    prompt_pairs: int


@dataclass
class LstmExampleStats:
    example_files: int
    vulnerability_types: Dict[str, int]


def extract_prompt_pairs(diff: str) -> int:
    prompt_pairs = 0
    prompt_start = diff.find("\n-")
    while prompt_start != -1:
        prompt_end = diff.find("\n+", prompt_start)
        label_end = diff.find("\n-", prompt_end)
        if label_end == -1 or (diff.find("\n ", prompt_end) < label_end):
            label_end = diff.find("\n ", prompt_end)
        prompt_pairs += 1
        prompt_start = diff.find("\n-", label_end)
    return prompt_pairs


def get_llm_dataset_stats(data_path: str) -> LlmDatasetStats:
    with open(data_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    repos = len(data)
    commits = 0
    files = 0
    prompt_pairs = 0

    for commits_info in data.values():
        commits += len(commits_info)
        for commit_info in commits_info.values():
            files_info = commit_info.get("files", {})
            files += len(files_info)
            for file_info in files_info.values():
                for change in file_info.get("changes", []):
                    diff = change.get("diff", "")
                    prompt_pairs += extract_prompt_pairs(diff)

    return LlmDatasetStats(
        repos=repos,
        commits=commits,
        files=files,
        prompt_pairs=prompt_pairs,
    )


def get_lstm_example_stats(examples_dir: str) -> LstmExampleStats:
    examples = [
        filename
        for filename in os.listdir(examples_dir)
        if filename.endswith(".py")
    ]
    vulnerability_types = Counter(
        filename.split("-")[0] for filename in examples
    )
    return LstmExampleStats(
        example_files=len(examples),
        vulnerability_types=dict(vulnerability_types),
    )


def format_vulnerability_types(vulnerability_types: Dict[str, int]) -> str:
    parts = [
        f"{name} ({count})"
        for name, count in sorted(vulnerability_types.items())
    ]
    return ", ".join(parts)


def render_markdown_table(llm_stats: LlmDatasetStats, lstm_stats: LstmExampleStats) -> str:
    types_summary = format_vulnerability_types(lstm_stats.vulnerability_types)
    return "\n".join(
        [
            "| Method | Source | Repos | Commits | Files | Prompt/label pairs | LSTM example files | LSTM vulnerability types |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| LLM-only repair | data/sql_injection.json | {llm_stats.repos} | {llm_stats.commits} | {llm_stats.files} | {llm_stats.prompt_pairs} | - | - |",
            f"| Git-Diff prompting repair | data/sql_injection.json | {llm_stats.repos} | {llm_stats.commits} | {llm_stats.files} | {llm_stats.prompt_pairs} | - | - |",
            f"| LLM repair for LSTM-detected vulnerabilities | lstm_examples/*.py | - | - | - | - | {lstm_stats.example_files} | {types_summary} |",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset counts for README tables.")
    parser.add_argument("--llm-data", default="data/sql_injection.json")
    parser.add_argument("--lstm-examples", default="lstm_examples")
    args = parser.parse_args()

    llm_stats = get_llm_dataset_stats(args.llm_data)
    lstm_stats = get_lstm_example_stats(args.lstm_examples)

    print(render_markdown_table(llm_stats, lstm_stats))


if __name__ == "__main__":
    main()
