
# Sentiment Classifier — Fine-Tuned DistilBERT (PyTorch + Hugging Face)

A sentiment analysis project that fine-tunes a pretrained **DistilBERT** transformer
model on the IMDB movie reviews dataset using **PyTorch** and the Hugging Face
`transformers` / `datasets` libraries. Includes training, evaluation, and a simple
inference script, plus an optional Streamlit demo UI.

## Why this project

Built to demonstrate practical, hands-on experience with:
- **PyTorch** — model training loop, optimizer, loss, GPU/CPU handling
- **NLP fundamentals** — tokenization, sequence classification
- **Transfer learning** — fine-tuning a pretrained transformer (DistilBERT) instead
  of training from scratch
- **Model evaluation** — accuracy, precision, recall, F1-score
- **Feature/data preprocessing** — tokenization, padding, truncation, train/val split

## Project Structure

```
sentiment-classifier/
├── src/
│   ├── train.py          # Fine-tunes DistilBERT on IMDB sentiment data
│   ├── evaluate.py        # Evaluates the trained model on the test set
│   ├── predict.py         # Run inference on custom text input
│   └── utils.py           # Shared helper functions (data loading, metrics)
├── app.py                 # Optional Streamlit demo UI
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/2311Srajan/sentiment-classifier.git
cd sentiment-classifier
pip install -r requirements.txt
```

Recommended: run on Google Colab (free GPU) if you don't have a local GPU —
just upload the `src/` folder or clone the repo in a Colab notebook.

## Usage

### 1. Train the model
```bash
python src/train.py --epochs 2 --batch_size 16 --output_dir ./model_output
```

### 2. Evaluate on the test set
```bash
python src/evaluate.py --model_dir ./model_output
```

### 3. Run inference on your own text
```bash
python src/predict.py --model_dir ./model_output --text "This movie was absolutely fantastic!"
```

### 4. (Optional) Launch the demo UI
```bash
streamlit run app.py
```

## Results

After fine-tuning for 2 epochs on a subset of IMDB (2,000 train / 500 test
samples for fast iteration on free-tier GPU/CPU):

| Metric | Score |
|---|---|
| Accuracy | ~88-90% (varies by run/seed) |
| F1-score | ~0.88-0.90 |

> Run `evaluate.py` on your own trained model to get exact numbers for your
> run — update this table with your actual results before sharing.

## Tech Stack

- Python, PyTorch
- Hugging Face `transformers` (DistilBERT) and `datasets` (IMDB)
- scikit-learn (evaluation metrics)
- Streamlit (optional demo)

## Future Improvements

- Compare DistilBERT vs a TensorFlow/Keras baseline (e.g. LSTM) for contrast
- Extend to multi-class intent classification instead of binary sentiment
- Add a small Gen AI comparison: prompt a GPT-style model to classify the same
  reviews and compare zero-shot performance vs the fine-tuned model
