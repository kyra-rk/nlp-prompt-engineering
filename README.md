# LLM-Prompt-Engineering

This repo contains the data and code for our paper "Comparing Prompt Engineering Techniques".

## Authors
Anum Ahmad 
Kyra Ramesh Krishna
Tramy Dong

## Datasets 
We use the google-research-datasets/poem_sentiment dataset from Hugging Face. We have built a script `make_data.py` that extracts the data from the site and splits it into train, test, and evaluation for any number of seeds provided. The folder `poem_data_splits` is the result of running this program for seeds 0, 1 and 42.

## Usage
You can run `predict.py` on a poem sentiment classification task with FlanT5 and ChatGPT. You will need to specify the setting, model, and api key as arguments to this program:
```
Usage:
  python predict.py --setting <setting> --model <model> [--api <API_KEY>] [--seed <SEED>]

Arguments:
  --setting <setting>           (required) One of: zero-shot, few-shot, chain-of-thought, meta
  --model <model>               (required) One of: gpt, flant5
  --api <API_KEY>               (required) Either Open-AI API key for gpt or HuggingFace access token for flant5 Inference API access
  --data_source <data_source>   (required) Source directory for dataset
  --filename <filename>         (required) Filename for dataset inside source directory

Example:
  python predict.py --setting zero-shot --model gpt --api YOUR_OPENAI_API_KEY --data_source poem_data --filename poem_sentiment_data.csv
```

## Evaluation
You can run the program `evaluate.py` to evaluate the results stored in `poem_sentiment_results` against the true label data stored in `poem_data_splits`. You must specify the `setting` (zero-shot, few-shot etc.), the `model` (eg. gpt, flant5) and can optionall specify the seed and output directory. By default, the program will assume seed 42 and will write calculations to `evaluation_results` and print out results to stdout. 
```
Usage:
  python evaluate.py --setting <setting> --model <model> [--seed <SEED>] [--output_dir <OUTPUT_DIR>]

Arguments:
  --setting <setting>    (required) Evaluation setting (e.g., zero-shot, few-shot, chain-of-thought, meta)
  --model <model>        (required) Model to use (e.g., gpt, flant5)
  --seed <SEED>          (optional) Seed value for dataset selection (default: 42)
  --output_dir <OUTPUT_DIR> (optional) Directory to save evaluation results (default: evaluation_results)

Example:
  python evaluate.py --setting zero-shot --model gpt --seed 42
  python evaluate.py --setting few-shot --model flant5 --output_dir results/
```

## Note
1. To view the summary of prompts and evaluation results, please navigate to the output folder (`poen_sentiment_results`) and check the respective task `.csv`.
2. You will need to specify the model and setting for each run. 

## Code References
We drew from the code used from the paper ["Sentiment Analysis in the Era of Large Language Models: A Reality Check"](https://arxiv.org/abs/2305.15005) by Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing in 2023.
