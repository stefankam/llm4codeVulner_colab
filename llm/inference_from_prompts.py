import argparse

import torch
from transformers import AutoTokenizer

from evaluator.io_utils import write_references_predictions
from utils import get_model, get_model_type


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle.readlines()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM inference on a prompts file and write references/predictions."
    )
    parser.add_argument("--prompts", required=True, help="Path to prompts.txt")
    parser.add_argument("--references", required=False, help="Optional references.txt to copy.")
    parser.add_argument("--output-dir", required=True, help="Output directory for references/predictions.")
    parser.add_argument("--model-name", default="Salesforce/codet5-small")
    parser.add_argument("--model-type", default="t5", choices=["t5", "casual", "auto"])
    parser.add_argument("--model-path", default=None, help="Optional tuned model path.")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_type = get_model_type(args.model_type)
    model = get_model(args.model_name, model_type, save_path=args.model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    prompts = read_lines(args.prompts)
    predictions = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(input_ids, max_new_tokens=48)
        predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))

    references = []
    if args.references:
        references = read_lines(args.references)
    else:
        references = [""] * len(predictions)

    write_references_predictions(
        references_path=f"{args.output_dir}/references.txt",
        predictions_path=f"{args.output_dir}/predictions.txt",
        references=references,
        predictions=predictions,
    )


if __name__ == "__main__":
    main()
