import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import AsyncOpenAI
import requests
import asyncio
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Classification Predictions")
    parser.add_argument("--setting", choices=["zero-shot", "few-shot", "chain-of-thought", "meta"], help="The type of prompt to use.")
    parser.add_argument("--model", choices=["gpt", "flant5"], help="The model to use.")
    parser.add_argument("--api", help="The OpenAI API key (only required for GPT models).")
    parser.add_argument("--data_source", help="Source directory containing the data file (e.g., poem_data).")
    parser.add_argument("--filename", help="Filename of the .csv file containing the data (e.g., data.csv).")
    return parser.parse_args()

# Load dataset from local files 
def load_local_dataset(data_source, filename):
    if not os.path.exists(data_source):
        raise ValueError(f"Source directory not found: {data_source}")

    test_path = os.path.join(data_source, f"{filename}")

    print(test_path)
    if not (os.path.exists(test_path)):
        raise ValueError(f"File {filename} in {data_source} not found.")
    
    test_df = pd.read_csv(test_path)

    return test_df

# Define the prompting functions
def zero_shot_prompt(text):
    return f"Classify the sentiment of the following text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)\n{text}"

def few_shot_prompt(text):
    examples = """
    Here are examples of sentiment classification:
    1. "and that is why, the lonesome day," -> negative
    2. "with pale blue berries. in these peaceful shades--" -> positive
    3. "it flows so long as falls the rain," -> no_impact
    4. "when i peruse the conquered fame of heroes, and the victories of mighty generals, i do not envy the generals," -> mixed
    """
    return f"{examples}\nClassify the following text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)\n{text}"

def chain_of_thought_prompt(text):
    examples = """
    Here are examples of sentiment classification and their reasonings:
    1. "and that is why, the lonesome day," -> negative because the use of "lonesome" suggests sadness or isolation, contributing to a negative sentiment.
    2. "with pale blue berries. in these peaceful shades--" -> positive because the description conveys tranquility and beauty, using calming imagery ("peaceful shades"), which evokes a positive emotional response.
    3. "it flows so long as falls the rain," -> no_impact because the text is neutral and descriptive, focusing on a natural process ("flows" and "rain") without emotional or evaluative language that indicates sentiment.
    4. "when i peruse the conquered fame of heroes, and the victories of mighty generals, i do not envy the generals," -> mixed because the text acknowledges achievements (positive) but lacks admiration or envy (negative), resulting in a mixed sentiment.
    """
    return f"{examples}\nClassify the following text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed) using reasoning:\n{text}"

def meta_prompt(text):
    return f"Text:{text}\nSolution Structure: 1. Think about and explain the emotional tone of the text 2. Think about and explain the context of the text 3. Based on tone and context, classify the text into one of the four categories: 0 (negative), 1 (positive), 2 (no_impact), 3 (mixed)"

# Retry logic for API calls
def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}")

@retry(wait=wait_fixed(10), stop=stop_after_attempt(6))
def query_chatgpt_model(api_key: str, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    try:
        # Make the POST request to the OpenAI API
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for HTTP issues
        
        # Parse the response JSON
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying model: {e}")
        raise

@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_flant5_model(api_key, prompt):
    model_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error querying Flan-T5 model: {e}")
        print(response.json())
    return pred


# Process the dataset and apply prompts
def process_dataset(dataset, output_folder, setting, model_query_function, api_key, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []

    for example in tqdm(dataset):
        text = example["text"]
        
        try:
            if setting == "zero-shot":
                prompt = zero_shot_prompt(text)
            elif setting == "few-shot":
                prompt = few_shot_prompt(text)
            elif setting == "chain-of-thought":
                prompt = chain_of_thought_prompt(text)
            elif setting == "meta":
                prompt = meta_prompt(text)
            else:
                raise ValueError("Invalid setting provided.")

            prediction = model_query_function(api_key, prompt)
        except Exception as e:
            print(f"Skipping due to repeated failures: {e}")
            prediction = "Error"

        results.append({
            "text": text,
            "prompt": prompt,
            "prediction": prediction
        })

    output_path = os.path.join(output_folder, f"predictions_{model}_{setting}.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Print usage instructions
def print_usage():
    print("""
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
""")

# Main function
def main():
    args = parse_args()

    # Validate arguments
    if not args.setting or not args.model:
        print("Error: Missing required arguments.")
        print_usage()
        return

    if not args.api:
        print("Error: API key required for gpt and flant5 models.")
        print_usage()
        return

    if not args.data_source or not args.filename:
        print("Error: data source and filename not provided,")
        print_usage()
        return

    try:
        test_df = load_local_dataset(args.data_source, args.filename)
        # for now we use test dataset since we are not training this model
        dataset = test_df.to_dict(orient="records")
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    name = os.path.splitext(args.filename)[0]
    output_folder = f"{args.data_source}_{name}_{args.model}_predictions"
    print(f"writing to {output_folder}")

    api_key = args.api
    model_query_function = query_chatgpt_model if args.model == "gpt" else query_flant5_model

    process_dataset(dataset, output_folder, args.setting, model_query_function, api_key, args.model)

if __name__ == "__main__":
    main()
