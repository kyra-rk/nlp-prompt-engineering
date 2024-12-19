import argparse
import os
import pandas as pd
from datasets import load_dataset

def load_dataset_from_huggingface(dataset_name, config_name=None):
    """
    Loads a dataset from Hugging Face with an optional configuration.
    """
    if config_name:
        print(f"Loading dataset '{dataset_name}' with config '{config_name}' from Hugging Face...")
        dataset = load_dataset(dataset_name, config_name)
    else:
        print(f"Loading dataset '{dataset_name}' without a config from Hugging Face...")
        dataset = load_dataset(dataset_name)
    return dataset

def combine_and_sample_dataset(dataset, output_dir, sample_size=1000):
    """
    Combines the 'train', 'validation', and 'test' splits into one DataFrame, saves the full data,
    and samples a subset of it.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine splits into one DataFrame
    combined_data = pd.concat([
        pd.DataFrame(dataset[split])
        for split in ["train", "validation", "test"]
        if split in dataset
    ], ignore_index=True)

    # Save the full dataset
    full_output_path = os.path.join(output_dir, "data.csv")
    combined_data.to_csv(full_output_path, index=False)
    print(f"Combined dataset saved to: {full_output_path}")

    # Randomly sample rows
    sampled_data = combined_data.sample(n=min(sample_size, len(combined_data)), random_state=42)
    sampled_output_path = os.path.join(output_dir, "truncated_data.csv")
    sampled_data.to_csv(sampled_output_path, index=False)
    print(f"Sampled dataset saved to: {sampled_output_path}")

def print_usage():
    print("""
Usage:
  python make_data.py --dataset_name <dataset_name> [--config_name <config_name>] --output_dir <output_directory> [--sample_size <sample_size>]

Arguments:
  --dataset_name <dataset_name>     (required) Hugging Face dataset name (e.g., 'tweet_eval').
  --config_name <config_name>       (optional) Configuration name if the dataset has multiple configurations.
  --output_dir <output_directory>   (required) Directory to save the combined dataset.
  --sample_size <sample_size>       (optional) Number of rows to sample for truncated_data.csv (default: 1000).

Examples:
  python make_data.py --dataset_name tweet_eval --config_name sentiment --output_dir data --sample_size 1000
  python make_data.py --dataset_name google-research-datasets/poem_sentiment --output_dir data
""")

def main():
    parser = argparse.ArgumentParser(description="Combine train/val/test splits of a Hugging Face dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset (e.g., 'tweet_eval').")
    parser.add_argument("--config_name", type=str, help="Configuration name for the dataset, if applicable.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the combined dataset.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of rows to sample for truncated_data.csv (default: 1000).")
    args = parser.parse_args()

    if not args.dataset_name or not args.output_dir:
        print("Error: --dataset_name and --output_dir arguments are required.")
        print_usage()
        return

    dataset = load_dataset_from_huggingface(args.dataset_name, args.config_name)
    combine_and_sample_dataset(dataset, args.output_dir, args.sample_size)

if __name__ == "__main__":
    main()
