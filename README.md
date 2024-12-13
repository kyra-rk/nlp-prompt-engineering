# LLM-Prompt-Engineering

This repo contains the data and code for our paper "Comparing Prompt Engineering Techniques".

## TODO clean up this section!! 
## Usage
0. fill in your OpenAI api key in the bash files under `script` folder. For example:
```
python predict.py \
--setting zero-shot \
--model chat \
--use_api \
--api #your api here
```

1. Run zero-shot and evaluate
```
bash script/run_zero_shot.sh
bash script/eval_zero_shot.sh
```

2. Run few-shot and evaluate
```
bash script/run_few_shot.sh
bash script/eval_few_shot.sh
```

## Note
1. To view the summary of prompts and evaluation results, please navigate to the output folder and check the respective task folder.
2. You can specify `--selected_tasks` and `--selected_datasets` to only run with certain tasks or datasets.


## Reference
We drew from the code used from the paper ["Sentiment Analysis in the Era of Large Language Models: A Reality Check"](https://arxiv.org/abs/2305.15005) by Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing in 2023.
