import argparse
import os
import torch
from transformers import AutoTokenizer
from utils import (convert_to_dataset, max_new_token_length,
                   text_column, label_column, ModelType, get_model,
                   print_metrics, get_prompt_prefix, get_git_diff_prompt_prefix,
                   inference_compare_code, get_model_type)
from data.process.utils import read_prompts_from_json
from evaluator.io_utils import write_references_predictions

def main():
    parser = argparse.ArgumentParser(
        prog='Evaluation Scripts',
        description='Evaluation Scripts to print out semantic scores for trained and untrained models.',
        epilog=''
    )

    parser.add_argument('-v', '--vulnerability', type=str, default='sql_injection',
                        help='vulnerability type need to be repaired, default is sql_injection')
    parser.add_argument('-l', '--lang', type=str, default='python',
                        help='programming language need to be repaired, default is python')
    parser.add_argument('-m', '--model_name', type=str, default='Salesforce/codet5-small',
                        help='model need to be trained, default is Salesforce/codet5-small')
    parser.add_argument('-t', '--model_type', type=str, default='t5',
                        help='model type needed to be tested or trained. '
                             'It will used to initialized tokenizer from huggingface, default is t5. Use causal for '
                             'casualLM')
    parser.add_argument('--data_usage', type=float, default=1.0,
                        help='Indicate how the total data usage should be used, default is 1.0')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Indicate the usage of selected data should be used for training, default is 0.6.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Indicate the usage of selected data should be used for validation, default is 0.2.')

    parser.add_argument('--prompt_style', type=str, default='llm_only',
                        choices=['llm_only', 'git_diff'],
                        help='Prompt style: llm_only (plain snippet) or git_diff (diff-aware prompt).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional output directory to write references.txt and predictions.txt.')
    args = parser.parse_args()
    model_name = args.model_name
    model_type = get_model_type(args.model_type)
    vulnerability = args.vulnerability
    lang = args.lang
    save_directory = "llm/models/{}".format(vulnerability + "-" + model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_usage_ratio = args.data_usage

    tuned_config = os.path.join(save_directory, "config.json")
    if os.path.exists(tuned_config):
        trained_model = get_model(model_name, model_type, save_path=save_directory, device=device)
    else:
        print(f"Trained model config not found at {tuned_config}. Falling back to base model.")
        trained_model = get_model(model_name, model_type, device=device)
    untrained_model = get_model(model_name, model_type=model_type, device=device)


    if args.prompt_style == 'git_diff':
        prompt_prefix = get_git_diff_prompt_prefix(vulnerability, lang)
    else:
        prompt_prefix = get_prompt_prefix(vulnerability, lang)
    data_file = "data/{}.json".format(vulnerability)

    data_file = "data/{}.json".format(vulnerability)

    prompts, labels = read_prompts_from_json(data_file)

    train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels,
                                                                         train_ratio=args.train_ratio,
                                                                         val_ratio=args.val_ratio,
                                                                         data_usage_ratio=data_usage_ratio)

    references = []
    predictions = []
    baseline_predictions = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for example in test_dataset:
        inference_compare_code(target_tokenizer=tokenizer, baseline_tokenizer=tokenizer,
                               prompt_prefix=prompt_prefix,
                               baseline_predictions=baseline_predictions,
                               prompt=example[text_column], label=example[label_column],
                               target_model=trained_model, baseline_model=untrained_model, device=device,
                               references=references, predictions=predictions,
                               max_new_tokens=max_new_token_length)

    print("##################" + "Train model output metrics" + "##################")

    print_metrics(references, predictions, lang)

    print("##################" + "Raw model output metrics" + "##################")

    print_metrics(references, baseline_predictions, lang)

    if args.output_dir:
        output_dir = args.output_dir
        write_references_predictions(
            references_path=f"{output_dir}/references.txt",
            predictions_path=f"{output_dir}/predictions.txt",
            references=references,
            predictions=predictions,
        )


if __name__ == "__main__":
    main()
