
# Enhancing Code Security Through LLM-Based Repair and LSTM-Detected Vulnerabilities

This repository is the official implementation of Enhancing Code Security Through LLM-Based Repair and LSTM-Detected Vulnerabilities

## Requirements

To install requirements:

We recommand to use conda to set up the environments. 

The training code need to runned with nvidia gpus but the inferences should be able to run with locally by downloading provided the pre-training models

All the data can be accessed from [Data Link ](https://www.dropbox.com/scl/fo/ybl9m2me6k3vcm5d6hwia/AJuAq3R1T9yfVD1HQfRgKh0?rlkey=dww32pwcdykxdjg5vwvfw823g&st=x3jrgd0o&dl=0). 

This included 

User need to download and move all the data from the data repos to directories with the same name and location under this repo 

```setup
git clone GITHUB_REPO_LINK
export PYTHONPATH=$PYTHONPATH:~/GITHUB_DIR_PATH
cd GITHUB_DIR/
conda env create -f environment/environment.yml
conda activate llm
```
All the training data stored as GITHUP API responses has been included under the data files.

Besides running the code at local or remote server, we also use the google-colab to quickly test our results with partial data. The colab scripts can be found at 

## Training

To train the LLM model(s) in the paper, run this command:

```train
huggingface-cli login
python llm/train.py --vulnerability [vulnerability want to trained with] --lang [programming language want to test, default is python] --epoches [training epoch] --model-name [the fine-tuned model name] --model-type [either model is T5 or casualLM]
```
However, besides the basic provided parameters in the command, there exist multiple parameters has been hard coded and tuned direclty in the code. 
For example, the lora parameters/training and validation data ratio etc. More details can be found at llm/train.py#L18


## Evaluation
The code-bert and codebleu score in Table 1 can directly got by the training scripts by enabling the evaluation (default).
We also provided a evaluation scripts which consumed the repair prediction and raw code as files instead of training and inferences from scratch.

All the code about caculate the codebleu score is cloned from https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD, we do appearciate this. 

To get the static analysis results, we provided a end-to-end scripts which will read the github patches in testing dataset, download python scripts by the github url in the patch, run the llm inferences with code diffs, combine the generated repairs and run the bandit tools to count the vulnerabilities.
The personal github token is required to download the raw files. 

```static analysis
python llm/static_analysis_test.py -v sql_injection -l python -t t5  -m Salesforce/codet5-small --github-token PERSIONAL_GITHUB_LINK
```
Also, we provided a simple scripts (llm/inferences.py) just doing the inferences by given prompts and model pathes for debugging purposes. 


We also provide the evaluation scripts to get the table-1 results with the provided model example in data files (need to move the models under llm files), just run the evaluation scripts 

```static analysis
python llm/evaluation.py -v sql_injection -l python -t t5 -m Salesforce/codet5-small
```

To run the LSTM only to detect the vulnerability showed as Fig 2, pleasue run 

```LSTM 
python demonstrate_labeled.py [vulnerability, like sql] [severity level, 4 is used for our results]
```

## Pre-trained Models

Lstm models and one pre-trained models has been included for testing purposes

We plan to upload all pre-trained LLM models to hugging-face later

## Results

Our fined model achieves the following performance on :

### 

|   Model name         |   Code Bleu     |  Code Bert F1  | Code Bert Precision  | Code Bert Recall  | 
| ---------------------|---------------- | -------------- |--------------------  | ------------------|
| Fine-tuend CodeT5    |     0.2344      |    0.8265      |         0.8265       |      0.7840       |
| Codegemma (benchmark)|     0.256       |    0.7173      |         0.7840       |      0.7912       |



### Repair dataset counts (computed from repository data)

Counts below were generated via `python scripts/calc_table_counts.py`.

| Method | Source | Repos | Commits | Files | Prompt/label pairs | LSTM example files | LSTM vulnerability types |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLM-only repair (LLM-only detection + repair; no Git-Diff prompting) | data/sql_injection.json | 336 | 406 | 667 | 2609 | - | - |
| Git-Diff prompting repair (LLM repair using Git-Diff prompts) | data/sql_injection.json | 336 | 406 | 667 | 2609 | - | - |
| LLM repair for LSTM-detected vulnerabilities (LSTM detection + LLM repair) | lstm_examples/*.py | - | - | - | - | 21 | command_injection (3), open_redirect (3), path_disclosure (3), remote_code_execution (3), sql (3), xsrf (3), xss (3) |

### Repair evaluation metrics (CodeBLEU / CodeBERT / Exact Match / Bandit)

Use the script below to compute metrics per repair method once you have the reference and prediction files for that run:

```metrics
python scripts/compute_metrics.py \
  --references path/to/references.txt \
  --predictions path/to/predictions.txt \
  --lang python \
  --bandit-path path/to/fixed/code
```

The script reports CodeBLEU, CodeBERT, exact match, and Bandit findings (if Bandit is installed).

| Method | CodeBLEU | CodeBERT Precision | CodeBERT Recall | CodeBERT F1 | Exact Match |
| --- | --- | --- | --- | --- | --- |
| LLM-only repair | TBD (requires references/predictions) | TBD | TBD | TBD | TBD |
| Git-Diff prompting repair | TBD (requires references/predictions) | TBD | TBD | TBD | TBD |
| LLM repair for LSTM-detected vulnerabilities | TBD (requires references/predictions) | TBD | TBD | TBD | TBD |

To generate per-method `references.txt` / `predictions.txt` outputs:

```llm-only
python llm/evaluation.py -v sql_injection -l python -t t5 -m Salesforce/codet5-small \
  --prompt_style llm_only --output_dir outputs/llm_only
```

```git-diff
python llm/evaluation.py -v sql_injection -l python -t t5 -m Salesforce/codet5-small \
  --prompt_style git_diff --output_dir outputs/git_diff
```

```lstm-detection
python lstm/export_detections.py \
  --examples-dir lstm_examples \
  --labels-dir data/lstm_labels \
  --output-dir data/lstm_detections
python llm/inference_from_prompts.py \
  --prompts data/lstm_detections/prompts.txt \
  --references data/lstm_detections/references.txt \
  --output-dir outputs/lstm_llm
```

Provide `data/lstm_labels` with fixed-code counterparts (matched by filename) to enable metric computation.


## Contributing

This Repo presents a comprehensive approach to enhancing code security through the integration of LSTM to detect/LLM to repair vulnerabilities in python
All content in this repository is licensed under the MIT licence
