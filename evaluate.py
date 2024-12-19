import os
import pandas as pd
import sys
import argparse
from sklearn.metrics import f1_score, classification_report

def print_usage():
    print("""
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
""")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against true labels."
    )
    parser.add_argument("--setting", type=str,
                        help="Evaluation setting (e.g., zero-shot, few-shot, chain-of-thought, meta)")
    parser.add_argument("--model", type=str,
                        help="Model to use (e.g., gpt, flant5)")
    parser.add_argument("--seed", type=int,
                        help="Seed value to load specific dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    return parser.parse_args()

def get_file_paths(seed, setting):
    base_pred_path = "poem_sentiment_results"
    base_true_labels_path = "poem_data_splits"
    
    predictions_file = os.path.join(base_pred_path, f"predictions_{setting}.csv")
    true_labels_file = os.path.join(base_true_labels_path, f"seed_{seed}", "test.csv")
    
    return predictions_file, true_labels_file

def process_predictions(predictions_file):
    df_pred = pd.read_csv(predictions_file)
    label_map = {"0": 0, "1":1, "2":2, "3":3, "negative": 0, "positive": 1, "no_impact": 2, "mixed": 3}
    df_pred['prediction'] = df_pred['prediction'].map(label_map)
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

def save_errors(df_true, df_pred, output_dir, setting, seed):
    merged_df = pd.merge(df_true, df_pred, on="text", how="inner")
    errors_df = merged_df[merged_df['label'] != merged_df['prediction']]
    output_file = os.path.join(output_dir, f"errors_{setting}_seed{seed}.csv")
    os.makedirs(output_dir, exist_ok=True)
    errors_df.to_csv(output_file, index=False)
    return errors_df

def evaluate(predictions_file, true_labels_file, output_dir, setting, seed):
    df_pred = process_predictions(predictions_file)
    df_true = process_true_labels(true_labels_file)
    assert len(df_true) == len(df_pred), "Mismatch between number of true labels and predictions"
    f1 = calculate_f1(df_true['label'], df_pred['prediction'])
    
    print("Classification Report:")
    print(classification_report(df_true['label'], df_pred['prediction'], zero_division=0))
    save_errors(df_true, df_pred, output_dir, setting, seed)
    return f1

def main():
    args = parse_args()
    
    valid_settings = ["zero-shot", "few-shot", "chain-of-thought", "meta"]
    valid_models = ["gpt", "flant5"]

    if (not args.setting) or (args.setting not in valid_settings):
        print(f"Error: Missing valid '--setting' argument.")
        print_usage()
        sys.exit(1)
    if (not args.model) or (args.model not in valid_models):
        print(f"Error: Missing valid '--model' argument.")
        print_usage()
        sys.exit(1)
        
    if args.seed is None:
        args.seed = 42
    if args.output_dir is None:
        args.output_dir = "evaluation_results"
    
    print(f"  Setting: {args.setting}")
    print(f"  Model: {args.model}")
    print(f"  Seed: {args.seed}")
    print(f"  Output Directory: {args.output_dir}")
    
    predictions_file, true_labels_file = get_file_paths(args.seed, args.setting)
    
    print(f"Evaluating predictions from {predictions_file} against true labels in {true_labels_file}...")
    f1 = evaluate(predictions_file, true_labels_file, args.output_dir, args.setting, args.seed)
    print(f"Evaluation complete. F1 Score: {f1:.4f}")
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
