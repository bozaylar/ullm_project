# -*- coding: utf-8 -*-
"""
GPT-5 Safe Completions Evaluation on XSTest Dataset

We evaluate GPT-5 responses on the XSTest safety benchmark dataset, which contains prompts designed to test AI safety mechanisms,
including homonyms, figurative language, and potentially harmful content.


Authors:  Bora Özaylar, Niclas Hergenröther, Mathias Neitzel
"""

import pandas as pd
import openai
from datasets import load_dataset





# ============================================================================
# CONFIGURATION
# ============================================================================

# Set the OpenAI API key here
API_KEY = "API_KEY"  # Replace with your actual API key

# Initialize OpenAI client
client = openai.OpenAI(api_key=API_KEY)






# ============================================================================
# LOAD DATASET
# ============================================================================

def load_data_from_huggingface():
    """
    Load XSTest dataset from HuggingFace
    The XSTest dataset contains 450 prompts.

    Returns:
        pandas DataFrame with columns: id, prompt, type, label, focus, note
    """
    dataset = load_dataset('Paul/XSTest', split="train")           #XSTest only consists of a train split
    df = pd.DataFrame(dataset)
    print("dataset loaded")
    return df
    
    
    
   
    

def load_data_from_csv(filepath='xstest_prompts.csv'):
    """
    Load XSTest dataset from a local CSV file

    Args:
        filepath: Path to CSV file (default: 'dataset/xstest_prompts.csv')

    Returns:
        pandas DataFrame with prompt data
    """
    df = pd.read_csv(filepath)
    print("dataset loaded")
    return df



# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_with_gpt5(df, model="gpt-5", output_file="output/gpt5_responses.csv"):
    """
    Evaluate all prompts with GPT-5 and save responses to gpt5_responses.csv file with same structure as dataset.

    Args:
        df: DataFrame with 'prompt' column
        model: OpenAI model to use
        output_file: Output CSV filename

    Returns:
        DataFrame with results
    """
    
    
    # Add column for GPT-5 responses
    df['completion'] = ''

    # Evaluate each prompt
    # Loop through each prompt in the dataset
    for idx in range(1):
        row = df.iloc[idx]

        # print current prompt number and preview
        print(f"[{idx+1}/{len(df)}] Processing: {row['prompt'][:80]}...")

        try:

            # Make API call to GPT-5
            # System message sets the assistant's behavior
            # User message contains the actual prompt to evaluate

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row['prompt']}
                ]
            )


            #Extract the text response and store it in the completion column
            df.at[idx, 'completion'] = response.choices[0].message.content
            print("Done")

        # Prevents the entire run from failing on one error if API call fails, stores the error message instead
        except Exception as e:
            print(f"Error: {e}")
            df.at[idx, 'completion'] = f"ERROR: {e}"

        # Small delay to avoid OpenAI's possible rate limits
        time.sleep(0.5)

        # Save progress every 50 prompts to prevent data loss in case the script crashes, we can resume from the last backup
        if (idx + 1) % 50 == 0:
            df[['id', 'type', 'prompt', 'completion']].to_csv(f'backup_{idx+1}.csv', index=False)
            print(f"Progress saved at {idx+1}")

    # Create the final output with only the columns we need from the original dataset
    df_result = df[['id', 'type', 'prompt', 'completion']]

    # Save final results to CSV
    df_result.to_csv(output_file, index=False)

    print(f"Evaluation complete. Saved to {output_file}")
    print(f"Total prompts processed: {len(df_result)}")

    return df_result



# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Load dataset
    # Option 1: From HuggingFace
    # df = load_data_from_huggingface()

    # Option 2: From CSV file (uncomment to use)
    df = load_data_from_csv('dataset/xstest_prompts.csv')

    # Run evaluation on all prompts
    # This will process each prompt and collect GPT-5 responses
    results = evaluate_with_gpt5(
        df=df,
        model="gpt-5",
        output_file="output/gpt5_responses.csv"
    )

    print("Done.")