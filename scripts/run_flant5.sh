#!/bin/bash

# Define the API key and the common parts of the command
API_KEY="YOUR_API_KEY"
DATA_SOURCE="poem_sentiment"
FILENAME="data_truncated.csv"

# List of settings and models to loop through
SETTINGS=("zero-shot" "few-shot" "chain-of-thought" "meta")
MODELS=("flan-t5-small" "flan-t5-large")

# Run the script for each setting and model combination
for SETTING in "${SETTINGS[@]}"
do
    for MODEL in "${MODELS[@]}"
    do
        echo "Running for setting: $SETTING, model: $MODEL"
        python3 predict.py --setting $SETTING --model $MODEL --api $API_KEY --data_source $DATA_SOURCE --filename $FILENAME
    done
done
