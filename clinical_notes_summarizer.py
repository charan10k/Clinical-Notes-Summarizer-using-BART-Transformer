# clinical_notes_summarizer.py

import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm  # For progress tracking

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def prepare_clinical_notes(data):
    # Format the clinical notes in a structured way
    data['clinical_notes'] = data.apply(lambda row: f"""
PatientID: {row.name + 1}
Sex: {row['sex']}
Race: {row['race']}
Date of First Visit: {row['Specimen date']}
Last Follow Up: {row['Date of Last Follow Up']}
Medical Condition: Stage {row['Stage']}
Status: {row['Dead or Alive']}
Further checkup: {'Required' if row['Event'] == 0 else 'Not Required - Deceased' if row['Dead or Alive'] == 'Dead' else 'Not Required'}
""", axis=1)
    return data['clinical_notes'].dropna().tolist()

def initialize_model(model_name):
    try:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.to('cuda')
        return tokenizer, model
    except Exception as e:
        raise Exception(f"Error initializing model: {str(e)}")

def summarize_text(text, tokenizer, model):
    try:
        prompt = "Summarize the following patient information maintaining the given structure: "
        inputs = tokenizer.encode(prompt + text, return_tensors="pt", max_length=1024, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        summary_ids = model.generate(
            inputs, 
            max_length=200,
            min_length=50,
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error summarizing text: {str(e)}")
        return ""

def main():
    try:
        model_name = 'facebook/bart-large-cnn'
        file_path = '/content/clinical.csv'  # Update this path as needed

        # Load and prepare data
        data = load_data(file_path)
        print("Data loaded successfully. Sample:")
        print(data.head())

        clinical_notes = prepare_clinical_notes(data)

        # Initialize model
        tokenizer, model = initialize_model(model_name)

        # Generate summaries with progress bar
        summaries = []
        for note in tqdm(clinical_notes, desc="Generating summaries"):
            summary = summarize_text(note, tokenizer, model)
            summaries.append(summary)

            print("\nOriginal Note:", note)
            print("Summary:", summary)
            print("-" * 80)

        # Save results
        summary_df = pd.DataFrame({'clinical_note': clinical_notes, 'summary': summaries})
        summary_df.to_csv('clinical_note_summaries.csv', index=False)
        print("Summaries saved successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()