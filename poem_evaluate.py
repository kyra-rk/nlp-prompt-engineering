import os
import pandas as pd
import argparse
from sklearn.metrics import f1_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sentiment prediction results")
    parser.add_argument("--seed", type=int, default=42, help="Seed number (e.g., 0, 1, 42)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory where evaluation results will be saved")
    return parser.parse_args()


def get_file_paths(seed):
    # Define paths based on the seed number
    base_pred_path = "poem_sentiment_results"
    base_true_labels_path = "poem_data_splits"

    # Set the correct predictions file (zero-shot for example, change this if needed)
    predictions_file = os.path.join(base_pred_path, f"predictions_zero-shot.csv")
    
    # Set the path for the true labels file based on the seed number
    true_labels_file = os.path.join(base_true_labels_path, f"seed_{seed}", "test.csv")
    
    return predictions_file, true_labels_file


def process_predictions(predictions_file):
    # Read predictions CSV
    df_pred = pd.read_csv(predictions_file)
    
    # Map string predictions to numeric values: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)
    label_map = {"negative": 0, "positive": 1, "no_impact": 2, "mixed": 3}
    df_pred['prediction'] = df_pred['prediction'].apply(lambda x: label_map.get(x, x))  # Ensure unknowns stay as is
    
    return df_pred


def process_true_labels(true_labels_file):
    # Read true labels CSV
    df_true = pd.read_csv(true_labels_file)
    
    # Ensure true labels are in numeric format
    df_true['label'] = pd.to_numeric(df_true['label'], errors='coerce')
    
    return df_true


def calculate_f1(true_labels, pred_labels):
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    return f1


def save_errors(df_true, df_pred, output_dir):
    # Merge true labels and predictions on 'text'
    merged_df = pd.merge(df_true, df_pred, on="text", how="inner")
    
    # Filter rows where true labels do not match predictions
    errors_df = merged_df[merged_df['label'] != merged_df['prediction']]
    
    # Save errors to a CSV file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    errors_df.to_csv(os.path.join(output_dir, "error_results.csv"), index=False)
    
    return errors_df


def evaluate(predictions_file, true_labels_file, output_dir):
    # Process predictions and true labels
    df_pred = process_predictions(predictions_file)
    df_true = process_true_labels(true_labels_file)
    
    # Make sure the number of rows match between the two datasets
    assert len(df_true) == len(df_pred), "Mismatch between number of true labels and predictions"
    
    # Calculate F1 score
    f1 = calculate_f1(df_true['label'], df_pred['prediction'])
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(df_true['label'], df_pred['prediction'], zero_division=0))
    
    # Save errors to file
    errors_df = save_errors(df_true, df_pred, output_dir)
    
    # Output F1 score
    print(f"F1 Score: {f1:.4f}")
    return f1


def main():
    args = parse_args()
    
    # Get the file paths based on the seed number
    predictions_file, true_labels_file = get_file_paths(args.seed)
    
    print(f"Evaluating predictions from {predictions_file} against true labels in {true_labels_file}...")
    
    # Run evaluation
    f1 = evaluate(predictions_file, true_labels_file, args.output_dir)
    
    print(f"Evaluation complete. F1 Score: {f1:.4f}")
    print(f"Error results saved in {args.output_dir}/error_results.csv")


if __name__ == "__main__":
    main()
