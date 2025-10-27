"""Main pipeline for sentiment analysis task.

This script demonstrates the sentiment analysis module integration
with the existing project architecture:
1. Load review data
2. Predict sentiments using the trained model
3. Generate statistics and visualizations
"""

import logging
import logging.config
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config.config import (
    RAW_DATA_FILE,
    FIGURE_DIR,
    LOGGING_CONFIG
)
from src.sentiment_analysis import SentimentAnalyzer


def setup_logging():
    """Configure logging for the pipeline."""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)


def main():
    """Run the complete sentiment analysis pipeline."""
    logger = setup_logging()
    logger.info("="*80)
    logger.info("SENTIMENT ANALYSIS PIPELINE - MangeTaMain")
    logger.info("="*80)
    
    try:
        # ====================================================================
        # 1. LOAD DATA
        # ====================================================================
        logger.info("\n[1/4] Loading review data...")
        
        # Check if RAW_interactions.csv exists
        interactions_file = Path("data") / "RAW_interactions.csv"
        if not interactions_file.exists():
            logger.error(f"Data file not found: {interactions_file}")
            logger.info("Please place RAW_interactions.csv in the data/ folder")
            return
        
        df_reviews = pd.read_csv(interactions_file)
        logger.info(f"Loaded {len(df_reviews):,} reviews")
        
        # Take a sample for demonstration (remove this for full processing)
        sample_size = min(1000, len(df_reviews))
        df_sample = df_reviews.sample(n=sample_size, random_state=42)
        logger.info(f"Processing sample of {sample_size:,} reviews for demonstration")
        
        # ====================================================================
        # 2. PREDICT SENTIMENTS
        # ====================================================================
        logger.info("\n[2/4] Predicting sentiments...")
        
        sentiments = []
        confidence_scores = []
        
        for idx, review in enumerate(df_sample['review'].values, 1):
            if idx % 100 == 0:
                logger.info(f"  Processed {idx}/{len(df_sample)} reviews...")
            
            try:
                label, score = SentimentAnalyzer.predict_sentiment(str(review))
                sentiments.append(label)
                confidence_scores.append(score)
            except Exception as e:
                logger.warning(f"Error processing review {idx}: {e}")
                sentiments.append("unknown")
                confidence_scores.append(0.0)
        
        df_sample['sentiment_predicted'] = sentiments
        df_sample['confidence_score'] = confidence_scores
        
        logger.info("✓ Sentiment prediction completed")
        
        # ====================================================================
        # 3. GENERATE STATISTICS
        # ====================================================================
        logger.info("\n[3/4] Generating statistics...")
        
        sentiment_counts = df_sample['sentiment_predicted'].value_counts()
        logger.info("\n📊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_sample)) * 100
            logger.info(f"  {sentiment}: {count:,} ({percentage:.1f}%)")
        
        avg_confidence = df_sample['confidence_score'].mean()
        logger.info(f"\n📈 Average Confidence Score: {avg_confidence:.2%}")
        
        # Sentiment by rating
        if 'rating' in df_sample.columns:
            logger.info("\n📊 Sentiment by Rating:")
            sentiment_by_rating = df_sample.groupby('rating')['sentiment_predicted'].value_counts()
            logger.info(sentiment_by_rating)
        
        # ====================================================================
        # 4. SAVE RESULTS
        # ====================================================================
        logger.info("\n[4/4] Saving results...")
        
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "sentiment_analysis_results.csv"
        df_sample.to_csv(output_file, index=False)
        logger.info(f"✓ Results saved to: {output_file}")
        
        # Save summary statistics
        summary_file = output_dir / "sentiment_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SENTIMENT ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total reviews analyzed: {len(df_sample):,}\n\n")
            f.write("Sentiment Distribution:\n")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(df_sample)) * 100
                f.write(f"  {sentiment}: {count:,} ({percentage:.1f}%)\n")
            f.write(f"\nAverage Confidence Score: {avg_confidence:.2%}\n")
        
        logger.info(f"✓ Summary saved to: {summary_file}")
        
        # ====================================================================
        # EXAMPLES
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("EXAMPLE PREDICTIONS")
        logger.info("="*80)
        
        examples = df_sample.head(5)
        for _, row in examples.iterrows():
            review_text = str(row['review'])[:100] + "..." if len(str(row['review'])) > 100 else str(row['review'])
            logger.info(f"\nReview: {review_text}")
            logger.info(f"Rating: {row.get('rating', 'N/A')} stars")
            logger.info(f"Predicted: {row['sentiment_predicted']} (confidence: {row['confidence_score']:.2%})")
        
        logger.info("\n" + "="*80)
        logger.info("✨ SENTIMENT ANALYSIS PIPELINE COMPLETED SUCCESSFULLY! ✨")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
