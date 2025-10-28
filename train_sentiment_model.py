"""
Sentiment Analysis Model Training Script
Converted from train_sentiment_analysis.ipynb
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

print("=" * 80)
print("SENTIMENT ANALYSIS MODEL TRAINING")
print("=" * 80)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  No GPU detected, using CPU (training will be slower)")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

data_path = Path(__file__).parent / "data" / "raw" / "RAW_interactions.csv"
print(f"\nLoading: {data_path}")

df_reviews = pd.read_csv(data_path)
print(f"✓ Loaded: {len(df_reviews):,} reviews")
print(f"  Columns: {df_reviews.columns.tolist()}")

# =============================================================================
# 2. PREPROCESS DATA
# =============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING DATA")
print("=" * 80)

# Clean data
df_clean = df_reviews.dropna(subset=['review', 'rating']).copy()
df_clean = df_clean[df_clean['review'].str.strip() != '']
print(f"\n✓ After cleaning: {len(df_clean):,} reviews ({len(df_clean)/len(df_reviews)*100:.1f}%)")

# Create sentiment labels from ratings
def rating_to_sentiment(rating):
    """Convert rating (1-5) to sentiment label (0-2)"""
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df_clean['sentiment'] = df_clean['rating'].apply(rating_to_sentiment)
df_clean['sentiment_label'] = df_clean['sentiment'].map({
    0: 'negative',
    1: 'neutral',
    2: 'positive'
})

print(f"\n📊 Sentiment distribution:")
sentiment_counts = df_clean['sentiment_label'].value_counts()
for label, count in sentiment_counts.items():
    print(f"   {label}: {count:,} ({count/len(df_clean)*100:.1f}%)")

# =============================================================================
# 3. SAMPLE DATA (for faster training)
# =============================================================================
print("\n" + "=" * 80)
print("SAMPLING DATA")
print("=" * 80)

SAMPLE_SIZE = 50000  # Adjust based on your resources
print(f"\nTarget sample size: {SAMPLE_SIZE:,}")

if len(df_clean) > SAMPLE_SIZE:
    # Stratified sampling
    df_sample = df_clean.groupby('sentiment', group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 3), random_state=42)
    ).reset_index(drop=True)
    print(f"✓ Sample created: {len(df_sample):,} reviews")
else:
    df_sample = df_clean.copy()
    print(f"✓ Using all data: {len(df_sample):,} reviews")

print(f"\n📊 Sample distribution:")
for label, count in df_sample['sentiment_label'].value_counts().items():
    print(f"   {label}: {count:,}")

# =============================================================================
# 4. SPLIT DATA
# =============================================================================
print("\n" + "=" * 80)
print("SPLITTING DATA")
print("=" * 80)

# 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(
    df_sample,
    test_size=0.3,
    random_state=42,
    stratify=df_sample['sentiment']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['sentiment']
)

print(f"\n✓ Data split:")
print(f"   Train:      {len(train_df):,} ({len(train_df)/len(df_sample)*100:.1f}%)")
print(f"   Validation: {len(val_df):,} ({len(val_df)/len(df_sample)*100:.1f}%)")
print(f"   Test:       {len(test_df):,} ({len(test_df)/len(df_sample)*100:.1f}%)")

# =============================================================================
# 5. CREATE PYTORCH DATASET
# =============================================================================
print("\n" + "=" * 80)
print("PREPARING TOKENIZER AND DATASETS")
print("=" * 80)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✓ Tokenizer loaded")

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""

    def __init__(self, reviews, sentiments, tokenizer, max_length=256):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]

        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

MAX_LENGTH = 256

train_dataset = SentimentDataset(
    train_df['review'].values,
    train_df['sentiment'].values,
    tokenizer,
    MAX_LENGTH
)

val_dataset = SentimentDataset(
    val_df['review'].values,
    val_df['sentiment'].values,
    tokenizer,
    MAX_LENGTH
)

test_dataset = SentimentDataset(
    test_df['review'].values,
    test_df['sentiment'].values,
    tokenizer,
    MAX_LENGTH
)

print(f"\n✓ Datasets created:")
print(f"   Train:      {len(train_dataset):,}")
print(f"   Validation: {len(val_dataset):,}")
print(f"   Test:       {len(test_dataset):,}")

# =============================================================================
# 6. LOAD MODEL
# =============================================================================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3  # negative, neutral, positive
)

model.to(device)
print(f"✓ Model loaded and moved to {device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 7. CONFIGURE TRAINING
# =============================================================================
print("\n" + "=" * 80)
print("CONFIGURING TRAINING")
print("=" * 80)

output_dir = Path(__file__).parent / "outputs" / "sentiment_model"
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,  # Reduced from 5 for faster training
    per_device_train_batch_size=16 if torch.cuda.is_available() else 8,
    per_device_eval_batch_size=32 if torch.cuda.is_available() else 16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(output_dir / 'logs'),
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("✓ Trainer configured:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size (train): {training_args.per_device_train_batch_size}")
print(f"   Batch size (eval): {training_args.per_device_eval_batch_size}")
print(f"   FP16: {training_args.fp16}")

# =============================================================================
# 8. TRAIN MODEL
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING MODEL")
print("=" * 80)

print("\n🚀 Starting training...")
print("This will take some time depending on your hardware...")

trainer.train()

print("\n✓ Training completed!")

# =============================================================================
# 9. EVALUATE ON TEST SET
# =============================================================================
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

test_results = trainer.evaluate(test_dataset)
print(f"\n📊 Test results:")
print(f"   Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"   Loss: {test_results['eval_loss']:.4f}")

# Detailed predictions
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

print(f"\n📋 Classification report:")
target_names = ['negative', 'neutral', 'positive']
print(classification_report(true_labels, preds, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Sentiment Analysis')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
print(f"✓ Confusion matrix saved: {output_dir / 'confusion_matrix.png'}")

# =============================================================================
# 10. SAVE MODEL
# =============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_save_dir = Path(__file__).parent / "outputs" / "models" / "sentiment_model_roberta"
model_save_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(model_save_dir))
tokenizer.save_pretrained(str(model_save_dir))

print(f"✓ Model saved to: {model_save_dir}")
print(f"  Files:")
print(f"    - config.json")
print(f"    - pytorch_model.bin")
print(f"    - tokenizer files")

# =============================================================================
# 11. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"""
Training Configuration:
  • Model: {MODEL_NAME}
  • Training samples: {len(train_dataset):,}
  • Validation samples: {len(val_dataset):,}
  • Test samples: {len(test_dataset):,}
  • Epochs: {training_args.num_train_epochs}
  • Device: {device}

Results:
  • Test Accuracy: {test_results['eval_accuracy']:.4f}
  • Test Loss: {test_results['eval_loss']:.4f}

Model saved to:
  {model_save_dir}

Next steps:
  1. Test model: python test_sentiment_predictions.py
  2. Use in Streamlit app
  3. Deploy to production
""")

print("\n🎉 Sentiment analysis model training completed successfully!")
print("=" * 80)
