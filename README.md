# Extractive Summarization with BERT + BiLSTM + Attention

This project implements an **Extractive Summarization** model designed to summarize text documents by identifying and selecting the most important sentences. It leverages **BERT** for contextual embeddings, a **BiLSTM** for sequence modeling, and an **Attention mechanism** to score sentence importance.

## Project Goal
To build a robust summarization model capable of handling messy inputs like study notes, lecture transcripts, and textbook excerpts.

## Key Features
- **BERT Embeddings**: Uses a pre-trained `bert-base-uncased` model (frozen) to generate rich contextual sentence embeddings.
- **BiLSTM Encoder**: A Bidirectional LSTM processes the BERT embeddings to capture forward and backward context between sentences.
- **Attention Mechanism**: Computes importance scores for each sentence to determine if it should be part of the summary.
- **Oracle Labeling**: Automatically generates training labels (0/1) based on ROUGE overlap with abstractive ground truth summaries.
- **Trigram Blocking**: Implements a redundancy removal technique during inference to prevent repeated phrases in the final summary.

## Datasets
The notebook attempts to load and combine the following datasets:
1. **Webis-TLDR-17**: a dataset of Reddit posts and their summaries.
2. **WikiHow**: a dataset of instructions and their summaries.

*Note: If these datasets are unavailable, the code automatically falls back to the **CNN/DailyMail** dataset.*

## Requirements
The project requires Python 3 and the following libraries:
- `torch` (with CUDA support recommended)
- `transformers`
- `datasets`
- `rouge-score`
- `nltk`
- `numpy`
- `pandas`
- `tqdm`

### Installation
You can install the dependencies directly within the notebook or via pip:
```bash
pip install datasets transformers rouge-score nltk torch torchvision torchaudio
```

## How to Run
1. Open the Jupyter Notebook: `Extractive Summarization with BERT + BiLSTM + Attention.ipynb`
2. Run the cells sequentially.
3. **Training**: The notebook will train the model for 4 epochs (configurable) and save the best model as `best_model_bert.pth`.
4. **Inference**: The final section demonstrates how to use the trained model to summarize a sample text, applying Trigram Blocking evaluations.

## Model Architecture
1. **Input**: Tokenized sentences.
2. **Embedding Layer**: BERT (`bert-base-uncased`), frozen.
3. **Encoder**: Bi-directional LSTM (`hidden_dim=128`).
4. **Attention**: Learnable weights to compute sentence relevance.
5. **Classifier**: Fully connected layer + Sigmoid activation to output a probability (0-1) for each sentence.

## Results
The notebook tracks training and validation loss, utilizing a scheduler to reduce the learning rate on plateaus. It outputs ROUGE-1 and ROUGE-L scores for evaluation.
