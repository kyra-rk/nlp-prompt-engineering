#!/bin/bash

# Define the common parts of the command
TRUE_LABELS_FILE="data_truncated.csv"

# List of settings and models to loop through
SETTINGS=("zero-shot" "few-shot" "chain-of-thought" "meta")
MODELS=("gpt-4o-mini" "gpt-3.5-turbo")

# Run the evaluation script for each setting and model combination
for SETTING in "${SETTINGS[@]}"
do
    for MODEL in "${MODELS[@]}"
    do
        PREDICTIONS_DIR="poem_sentiment_data_truncated_${MODEL}_predictions"
	PREDICTIONS_FILE="predictions_${MODEL}_${SETTING}.csv"
        echo "Running evaluation for setting: $SETTING, model: $MODEL"
        python3 evaluate.py --task poem --setting $SETTING --model $MODEL --predictions_dir "${PREDICTIONS_DIR}" --predictions_file "${PREDICTIONS_FILE}" --true_labels_file $TRUE_LABELS_FILE
    done
done
