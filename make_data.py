import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_poem_sentiment_dataset():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    return dataset  # Now returning the entire dataset (train, validation, test)

def create_data_splits(output_dir, seeds):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the entire dataset (train, validation, test)
    dataset = load_poem_sentiment_dataset()

    # Combine the 'train', 'validation', and 'test' splits into one dataframe
    data = pd.concat([
        pd.DataFrame({"text": [example["verse_text"] for example in dataset["train"]], "label": [example["label"] for example in dataset["train"]]}),
        pd.DataFrame({"text": [example["verse_text"] for example in dataset["validation"]], "label": [example["label"] for example in dataset["validation"]]}),
        pd.DataFrame({"text": [example["verse_text"] for example in dataset["test"]], "label": [example["label"] for example in dataset["test"]]})
    ], ignore_index=True)

    for seed in seeds:
        print(f"Creating splits with seed {seed}...")
        # First, split the dataset into train (60%) and temp (40%)
        train, temp = train_test_split(data, test_size=0.4, random_state=seed)
        
        # Then split the temp data into validation (50%) and test (50%)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)

        # Create a directory for the current seed
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        # Save the splits as CSV files in the seed-specific directory
        train.to_csv(os.path.join(seed_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(seed_dir, "eval.csv"), index=False)
        test.to_csv(os.path.join(seed_dir, "test.csv"), index=False)

        print(f"Data splits for seed {seed} saved in {seed_dir}.")

def print_usage():
    print("""
Usage:
  python make_data.py --output_dir <output_directory> --seeds <seed1> <seed2> ...

Arguments:
  --output_dir <output_directory>   (required) Directory to save the data splits.
  --seeds <seed1> <seed2> ...       (required) One or more seeds for generating splits.

Example:
  python make_data.py --output_dir data_splits --seeds 0 1 42
""")

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for the Poem Sentiment dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the data splits.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="One or more seeds for generating splits.")
    args = parser.parse_args()

    if not args.output_dir:
        print("Error: --output_dir argument is required.")
        print_usage()
        return

    if not args.seeds:
        print("Error: --seeds argument is required and should include at least one seed.")
        print_usage()
        return

    create_data_splits(args.output_dir, args.seeds)

if __name__ == "__main__":
    main()