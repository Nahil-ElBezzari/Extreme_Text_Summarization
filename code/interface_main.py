import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Charger le modèle fine-tuné et le tokenizer
def load_model(model_dir="./t5_xsum_finetuned"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

tokenizer, model = load_model()

# Fonction de résumé
def summarize(text, max_input_length=512, max_output_length=64):
    input_text = "summarize: " + text.strip()
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)

    # Déplacer sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = inputs.to(device)

    # Génération du résumé
    summary_ids = model.generate(
        inputs,
        max_length=max_output_length,
        min_length=10,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Interface Gradio
interface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Entrer the text to summarize here...", label="Text"),
    outputs=gr.Textbox(label="Summary generated"),
    title="📝 Extreme text summarizer",
    description="Enter an article or a paragraph. The model will generate a concise summary.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()