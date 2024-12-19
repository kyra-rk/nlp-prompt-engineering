import os
import pandas as pd
import sys
import re
import argparse
from sklearn.metrics import f1_score, classification_report

def print_usage():
    print("""
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
    python3 evaluate.py --task poem --setting zero-shot --model flant5 --predictions_dir poem_sentiment_data_truncated_flant5_predictions --true_labels_file data_truncated.csv
""")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against true labels."
    )
    parser.add_argument("--task", type=str,
                        help="Sentiment classification task (choose from: poem, tweet, yelp)")
    parser.add_argument("--setting", type=str,
                        help="Evaluation setting (e.g., zero-shot, few-shot, chain-of-thought, meta)")
    parser.add_argument("--model", type=str,
                        help="Model to use (e.g., gpt, flant5)")
    parser.add_argument("--predictions_dir", type=str,
                        help="Directory to find prediction data in (default: <task>_sentiment_data_<model>_predictions)")
    parser.add_argument("--predictions_file", type=str,
                        help="Filename of prediction data (default: predictions_<model>_<setting>.csv)")
    parser.add_argument("--true_labels_dir", type=str,
                        help="Directory to find true label data in (default: <task>_sentiment)")
    parser.add_argument("--true_labels_file", type=str,
                        help="Filename of true label data (default: data.csv)")
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save evaluation results (default: <model>_<task>_evaluation_results)")
    return parser.parse_args()

def get_file_paths(predictions_dir, predictions_filename, true_label_dir, true_label_filename):
    predictions_file = os.path.join(predictions_dir, f"{predictions_filename}")
    true_labels_file = os.path.join(true_label_dir, f"{true_label_filename}")
    
    return predictions_file, true_labels_file

def process_predictions(predictions_file, task, true_labels_file=None):
    df_pred = pd.read_csv(predictions_file)
    
    # Load true labels if needed for validation
    if true_labels_file:
        df_true = pd.read_csv(true_labels_file)
    
    # Define label mappings for each task
    if task == "poem":
        label_map = {"0": 0, "1": 1, "2": 2, "3": 3, "negative": 0, "positive": 1, "no_impact": 2, "mixed": 3}
    elif task == "tweet":
        label_map = {"0": 0, "1": 1, "2": 2, "negative": 0, "neutral": 1, "positive": 2}
    elif task == "yelp":
        label_map = {"0": 0, "1": 1, "negative": 0, "positive": 1}
    else:
        raise ValueError("Unknown task specified.")
    
    def map_prediction(prediction):
        # If the prediction is already a number in string format
        if prediction in label_map:
            return label_map[prediction]
        # If the prediction is a single word
        if isinstance(prediction, str) and prediction.lower() in label_map:
            return label_map[prediction.lower()]
        # If the prediction is a full sentence, extract the number
        match = re.search(r'\b\d\b', str(prediction))
        if match:
            return int(match.group(0))
        # If no match is found, return None
        return None

    # Process the predictions column
    df_pred['prediction'] = df_pred['prediction'].apply(map_prediction)

    # Drop or handle NaN values
    df_pred.dropna(subset=['prediction'], inplace=True)

    return df_pred

def process_true_labels(true_labels_file):
    df_true = pd.read_csv(true_labels_file)
    df_true['label'] = pd.to_numeric(df_true['label'], errors='coerce')
    return df_true

def calculate_f1(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels, average='macro', zero_division=0)

def calculate_guessed(df_pred):
    # Calculate the number of guesses (predicted labels) for each label
    guessed_counts = df_pred['prediction'].value_counts().to_dict()
    return guessed_counts

def save_errors(df_true, df_pred, output_dir, setting, model):
    merged_df = pd.merge(df_true, df_pred, on="text", how="inner")
    errors_df = merged_df[merged_df['label'] != merged_df['prediction']]
    output_file = os.path.join(output_dir, f"errors_{model}_{setting}.csv")
    os.makedirs(output_dir, exist_ok=True)
    errors_df.to_csv(output_file, index=False)
    return errors_df

def evaluate(predictions_file, true_labels_file, output_dir, setting, model, task):
    df_pred = process_predictions(predictions_file, task)
    df_true = process_true_labels(true_labels_file)
    
    min_len = min(len(df_true), len(df_pred))
    f1 = calculate_f1(df_true['label'][:min_len], df_pred['prediction'][:min_len])
    
    print("Classification Report:")
    print(classification_report(df_true['label'][:min_len], df_pred['prediction'][:min_len], zero_division=0))
    save_errors(df_true[:min_len], df_pred[:min_len], output_dir, setting, model)
    return f1

def main():
    args = parse_args()
    
    valid_settings = ["zero-shot", "few-shot", "chain-of-thought", "meta"]
    valid_models = ["gpt", "flant5"]
    valid_tasks = ["poem", "tweet", "yelp"]

    if (not args.setting) or (args.setting not in valid_settings):
        print(f"Error: Missing valid '--setting' argument.")
        print_usage()
        sys.exit(1)
    if (not args.model) or (args.model not in valid_models):
        print(f"Error: Missing valid '--model' argument.")
        print_usage()
        sys.exit(1)
    if (not args.task) or (args.task not in valid_tasks):
        print(f"Error: Missing valid '--task' argument.")
        print_usage()
        sys.exit(1)
  
    if args.predictions_dir is None:
        args.predictions_dir = f"{args.task}_sentiment_data_{args.model}_predictions"
    if args.predictions_file is None:
        args.predictions_file = f"predictions_{args.model}_{args.setting}.csv"
    if args.true_labels_dir is None:
        args.true_labels_dir = f"{args.task}_sentiment"
    if args.true_labels_file is None:
        args.true_labels_file = "data.csv"
    if args.output_dir is None:
        args.output_dir = f"{args.model}_{args.task}_evaluation_results"
    
    print(f"  Task: {args.task}")
    print(f"  Setting: {args.setting}")
    print(f"  Model: {args.model}")
    print(f"  Predictions Source Directory: {args.predictions_dir}")
    print(f"  Predictions Source File: {args.predictions_file}")
    print(f"  True Labels Source Directory: {args.true_labels_dir}")
    print(f"  True Labels Source File: {args.true_labels_file}")
    print(f"  Output Directory: {args.output_dir}")
    
    predictions_file, true_labels_file = get_file_paths(args.predictions_dir, args.predictions_file, args.true_labels_dir, args.true_labels_file)
    
    print(f"Evaluating predictions from {predictions_file} against true labels in {true_labels_file}...")
    f1 = evaluate(predictions_file, true_labels_file, args.output_dir, args.setting, args.model, args.task)
    print(f"Evaluation complete. F1 Score: {f1:.4f}")
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
