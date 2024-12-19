# LLM-Prompt-Engineering

This repo contains the data and code for our paper "Comparing Prompt Engineering Techniques".

## Authors
Anum Ahmad, Kyra Ramesh Krishna, and Tramy Dong

## Datasets 
We use the google-research-datasets/poem_sentiment dataset from Hugging Face. We have built a script `make_data.py` that extracts the data from the site and combines the train, validation and test data into one cohesive dataset. The folder `poem_data` with the file `poem_sentiment_data.csv` is the result of running this program.

## Setting Up
We have conveniently created `setup_venv.sh` to help you set up a virtual environment that will install all the neccesary packages for you. Simply run `source setup_venv.sh`.

## Usage
You can run `predict.py` on a poem sentiment classification task with FlanT5 and ChatGPT. You will need to specify the setting, model, and api key as arguments to this program:
```
Usage:
  python3 predict.py --setting <setting> --model <model> [--api <API_KEY>] [--seed <SEED>]

Arguments:
  --setting <setting>           (required) One of: zero-shot, few-shot, chain-of-thought, meta
  --model <model>               (required) One of: gpt, flant5
  --api <API_KEY>               (required) Either Open-AI API key for gpt or HuggingFace access token for flant5 Inference API access
  --data_source <data_source>   (required) Source directory for dataset
  --filename <filename>         (required) Filename for dataset inside source directory

Example:
  python3 predict.py --setting zero-shot --model flant5 --api YOUR_HUGGINGFACE_ACCESS_TOKEN --data_source poem_sentiment --filename data.csv
```

## Evaluation
You can run the program `evaluate.py` to evaluate the results stored in `<task>_sentiment_data_<model>_predictions/predictions_<model>_<setting>.csv` against the true label data stored in `poem_sentiment/data.csv`. You must specify the `task` (one of: poem, tweet and yelp), the `setting` (one of: zero-shot, few-shot etc.), the `model` (one of: gpt, flant5). You can optionally specify the predictions file and directory, the true labels file and directory and the output directory. By default, the program will generate the file names based on the task, model and setting. Error results will be written to `<model>_<setting>_<task>_evaluation_results` and overall metric calculations will be printed to stdout. 
```
Usage:
  python3 evaluate.py --task <task> --setting <setting> --model <model> [--predictions_dir <predictions_dir>] [--predictions_file <predictions_file>] [--true_labels_dir <true_labels_dir>] [--true_labels_file <true_labels_file>] [--output_dir <output_dir>]

Arguments:
  --task <task>                         (required) Sentiment classification task (choose from: poem, tweet, yelp)
  --setting <setting>                   (required) Evaluation setting (e.g., zero-shot, few-shot, chain-of-thought, meta)
  --model <model>                       (required) Model to use (choose from: gpt, flant5)
  --predictions_dir <predictions_dir>   (optional) Directory to find prediction data in (default: <task>_sentiment_data_<model>_predictions)
  --predictions_file <predictions_file> (optional) Filename of prediction data (default: predictions_<model>_<setting>.csv)
  --true_labels_dir <true_labels_dir>   (optional) Directory to find true label data in (default: <task>_sentiment)
  --true_labels_file <true_labels_file> (optional) Filename of true label data (default: data.csv)
  --output_dir <output_dir>             (optional) Directory to save evaluation results (default: <model>_<task>_evaluation_results)

Example:
    python3 evaluate.py --setting zero-shot --model flant5 --true_labels_file data_truncated.csv
```

## Note
1. To view the summary of prompts and evaluation results, please navigate to the output folder (eg. `poen_sentiment_results`) and check the respective setting `.csv` (eg. zero-shot, few-shot etc.).
2. You will need to specify the model and setting for each run. 
3. The `predict.py` and `evaluate.py` code can be generalized for many other classification tasks as well. Just note to update the arguments and the prediction extraction.
4. You can find sample results from `evaluate.py` on poem sentiment classification in `project_eval_results.txt`, which are the results after running on a truncated portion of the poem dataset.

## Code References
We drew from the code used from the paper ["Sentiment Analysis in the Era of Large Language Models: A Reality Check"](https://arxiv.org/abs/2305.15005) by Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing in 2023.
