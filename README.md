# 📝 Extreme Text Summarization

**Extreme Text Summarization** is a Natural Language Processing (NLP) project designed to generate concise and informative one-line summaries ("headlines") from longer texts. It leverages modern Transformer-based architectures to produce high-quality, abstractive summaries suitable for applications like article title generation and fast content previews.

---

## 🚀 Features

- ✅ **Interactive Web Interface** using Gradio for easy real-time summarization
- ✅ **Command-line Version** for environments without a GUI
- ✅ **Jupyter Notebooks** with multiple model training experiments

---

## 🎯 Project Goal

The goal of this project is to build an automated summarization system that can condense a long piece of text into a meaningful, single-sentence summary. This is particularly useful for news headlines, email previews, or content recommendation systems.

---

## 📁 Project Structure

```
.
├── interface_main.py        # Gradio interface for demo
├── main.py                  # Command-line interface
├── models_notebook/         # Notebooks for training and experimentation
│   ├── LSTM_model.ipynb
│   ├── T5_Fine_tuning.ipynb
│   └── Transformers_model.ipynb
├── t5_xsum_finetuned/       # Google T5 model finetned on the XSum dataset
```

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Nahil-ElBezzari/Extreme_Text_Summarization.git
cd Extreme_Text_Summarization
```

2. **Install dependencies:**

```bash
pip install transformers torch gradio
```

---

## 🧪 How to Use

### Option 1: Web Interface

Run the interactive Gradio app:

```bash
python interface_main.py
```

A local web page will open where you can input text and get summaries in real time.

### Option 2: Command-Line Interface

Run the CLI version:

```bash
python main.py
```

Follow the prompts to enter your text and view the generated summary.

---

## 📊 Model Training

The `models_notebook/` folder contains Jupyter notebooks with step-by-step model training and evaluation. These include:

- Data preprocessing
- Fine-tuning pre-trained Transformer models
- ROUGE score evaluation

You can open and explore these notebooks with Jupyter Lab or Google Colab.

---

## 📈 Results

Model performance is evaluated using ROUGE metrics. Results vary depending on the model architecture and training parameters. Please refer to the individual notebooks for detailed scores and model comparisons.

---
