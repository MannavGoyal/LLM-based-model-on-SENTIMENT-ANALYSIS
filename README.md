# Sentiment Analysis

A complete sentiment analysis system using DistilBERT for classifying text into positive, negative, or neutral sentiments.

## Features

- **DistilBERT-based**: Uses the efficient DistilBERT model for fast and accurate sentiment classification
- **3-class Classification**: Classifies text into positive, negative, or neutral sentiments
- **Easy Training**: Simple training pipeline with automatic model saving
- **Interactive Inference**: Command-line interface for testing the model
- **Comprehensive Evaluation**: Detailed metrics, plots, and confusion matrices
- **GPU Support**: Automatic GPU detection and usage

## Project Structure

```
sentiment_analysis/
├── config.py              # Configuration settings
├── model.py               # DistilBERT model definition
├── data_loader.py         # Data loading and preprocessing
├── trainer.py             # Training logic and evaluation
├── train.py               # Main training script
├── inference.py           # Inference and demo scripts
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data directory
├── models/               # Saved models
└── results/              # Training results and plots
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify PyTorch installation**:
   ```python
   import torch
   print(torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   ```

## Quick Start

### 1. Training the Model

Run the training script:

```bash
python train.py
```

This will:
- Download the DistilBERT model
- Load sample data (or IMDB dataset if available)
- Train the model for 3 epochs
- Save the best model to `models/best_model.pth`
- Generate training plots and metrics in `results/`

### 2. Testing the Model

**Interactive mode**:
```bash
python inference.py --interactive
```

**Single text prediction**:
```bash
python inference.py --text "I love this product!"
```

**Demo examples**:
```bash
python inference.py --demo
```

## Usage Examples

### Training with Custom Data

1. **Prepare your data** in CSV format with columns:
   - `text`: The text to classify
   - `sentiment`: Labels (0=negative, 1=neutral, 2=positive)

2. **Modify `data_loader.py`** to load your custom dataset:
   ```python
   def load_custom_data(self, file_path):
       df = pd.read_csv(file_path)
       return df
   ```

3. **Update `train.py`** to use your data:
   ```python
   train_df = data_loader.load_custom_data("your_data.csv")
   ```

### Inference Examples

```python
from inference import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor("models/best_model.pth")

# Single prediction
result = predictor.predict_single("This movie is fantastic!")
print(result)

# Batch prediction
texts = ["Great product!", "Terrible service", "It's okay"]
results = predictor.predict_batch(texts)
for result in results:
    print(f"{result['text']} -> {result['predicted_sentiment']}")
```

## Configuration

Modify `config.py` to customize:

- **Model settings**: Model name, max sequence length
- **Training parameters**: Learning rate, batch size, epochs
- **Data splits**: Train/validation/test ratios
- **Paths**: Data, model, and results directories

```python
class Config:
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    # ... more settings
```

## Model Performance

The model typically achieves:
- **Training time**: ~10-30 minutes (depending on data size and hardware)
- **Accuracy**: 85-95% on validation data
- **Inference speed**: ~100-1000 texts per second (GPU)

## Advanced Usage

### Custom Model Architecture

Modify `model.py` to customize the model:

```python
class CustomDistilBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, Config.NUM_LABELS)
        # Add more layers as needed
```

### Training Callbacks

Add custom callbacks in `trainer.py`:

```python
def on_epoch_end(self, epoch, logs):
    # Custom logic after each epoch
    if logs['val_accuracy'] > 0.95:
        print("Early stopping - target accuracy reached!")
        return True  # Stop training
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `BATCH_SIZE` in `config.py`
   - Reduce `MAX_LENGTH` for shorter sequences

2. **Slow training**:
   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Use smaller model: `distilbert-base-uncased` instead of `bert-base-uncased`

3. **Poor accuracy**:
   - Increase training epochs
   - Use more training data
   - Adjust learning rate
   - Check data quality and labels

### Performance Optimization

1. **Use mixed precision training**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

2. **Gradient accumulation** for larger effective batch size:
   ```python
   accumulation_steps = 4
   if (step + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

## Dataset Information

### Default Datasets

1. **Sample Data**: 10 example texts for quick testing
2. **IMDB Dataset**: 50k movie reviews (automatically downloaded)

### Custom Dataset Format

Your CSV should have these columns:
```csv
text,sentiment
"I love this product!",2
"This is terrible.",0
"It's okay I guess.",1
```

Labels:
- `0`: Negative
- `1`: Neutral  
- `2`: Positive

## API Reference

### SentimentPredictor

```python
predictor = SentimentPredictor(model_path="models/best_model.pth")

# Single prediction
result = predictor.predict_single(text)
# Returns: {'text': str, 'predicted_sentiment': str, 'confidence': float, 'probabilities': dict}

# Batch prediction
results = predictor.predict_batch(texts)
# Returns: List of prediction dictionaries
```

### Training

```python
from trainer import SentimentTrainer

trainer = SentimentTrainer(model, train_loader, val_loader)
trainer.train()  # Start training
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Hugging Face Transformers library
- DistilBERT paper: "DistilBERT, a distilled version of BERT"
- PyTorch team for the deep learning framework
