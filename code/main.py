from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model(model_dir="./t5_xsum_finetuned"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

def summarize(text, tokenizer, model, max_input_length=512, max_output_length=64):
    # Preprocessing the input
    input_text = "summarize: " + text.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)

    # Generating resume
    summary_ids = model.generate(inputs, max_length=max_output_length, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    # Loading the model and tokenizer
    tokenizer, model = load_model()

    print("=== T5 Text Summarizer ===")
    print("Enter a text to summarize:")

    try:
        while True:
            input_text = input("\nYour text :\n")
            if not input_text.strip():
                print("Enter a non-empty text.")
                continue
            summary = summarize(input_text, tokenizer, model)
            print("\nGenerated summary :")
            print(summary)
    except KeyboardInterrupt:
        print("\nEnd program.")

if __name__ == "__main__":
    main()