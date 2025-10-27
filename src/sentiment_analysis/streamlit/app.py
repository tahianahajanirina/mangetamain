"""
Streamlit Web App for Recipe Review Sentiment Analysis

This app allows users to:
1. Analyze sentiment of individual recipe reviews
2. Batch analyze multiple reviews
3. View sentiment statistics

Run with: streamlit run src/sentiment_analysis/streamlit/app.py
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.sentiment_analysis import SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Recipe Review Sentiment Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🍽️ Recipe Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.info("""
    This app uses a fine-tuned RoBERTa model to analyze sentiments in food recipe reviews.
    
    **Model:** TahianaAndriambahoaka/sentiment-analysis-food-reviews
    
    **Sentiments:**
    - 🟢 Positive
    - 🟡 Neutral
    - 🔴 Negative
    """)
    
    st.header("🔧 Options")
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_examples = st.checkbox("Show example reviews", value=True)

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📋 Batch Analysis", "📈 Statistics"])

# Tab 1: Single Review Analysis
with tab1:
    st.header("Analyze a Single Review")
    
    if show_examples:
        st.subheader("Example Reviews (click to use)")
        examples = {
            "Positive": "This recipe was absolutely amazing! I will definitely make it again. My family loved it!",
            "Negative": "Too salty and the instructions were confusing. Waste of time and ingredients.",
            "Neutral": "It was okay, nothing special. Average taste, easy to make."
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🟢 Use Positive Example"):
                st.session_state.review_text = examples["Positive"]
        with col2:
            if st.button("🟡 Use Neutral Example"):
                st.session_state.review_text = examples["Neutral"]
        with col3:
            if st.button("🔴 Use Negative Example"):
                st.session_state.review_text = examples["Negative"]
    
    # Text input
    review_text = st.text_area(
        "Enter a recipe review:",
        value=st.session_state.get('review_text', ''),
        height=150,
        placeholder="Type or paste a recipe review here..."
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    label, confidence = SentimentAnalyzer.predict_sentiment(review_text)
                    
                    # Display result
                    st.markdown("### Result")
                    
                    # Determine sentiment class and emoji
                    if label.lower() in ['positive', 'label_2']:
                        sentiment_class = "sentiment-positive"
                        emoji = "🟢"
                        sentiment_text = "Positive"
                        color = "#28a745"
                    elif label.lower() in ['negative', 'label_0']:
                        sentiment_class = "sentiment-negative"
                        emoji = "🔴"
                        sentiment_text = "Negative"
                        color = "#dc3545"
                    else:
                        sentiment_class = "sentiment-neutral"
                        emoji = "🟡"
                        sentiment_text = "Neutral"
                        color = "#ffc107"
                    
                    # Display sentiment box
                    st.markdown(f"""
                        <div class="{sentiment_class}">
                            <h2 style="margin:0; color:{color};">{emoji} {sentiment_text}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence
                    if show_confidence:
                        st.markdown("### Confidence Score")
                        st.progress(confidence)
                        st.markdown(f'<p class="confidence-score">{confidence:.1%}</p>', unsafe_allow_html=True)
                    
                    # Display review
                    with st.expander("📄 Review Text"):
                        st.write(review_text)
                    
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e}")
        else:
            st.warning("Please enter a review to analyze.")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Analysis")
    st.write("Analyze multiple reviews at once")
    
    # Option 1: Upload CSV
    st.subheader("Option 1: Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'review' not in df.columns:
                st.error("CSV must contain a 'review' column")
            else:
                st.success(f"Loaded {len(df)} reviews")
                
                if st.button("🔍 Analyze All Reviews"):
                    with st.spinner("Analyzing reviews..."):
                        sentiments = []
                        confidences = []
                        
                        progress_bar = st.progress(0)
                        for idx, review in enumerate(df['review'].values):
                            try:
                                label, confidence = SentimentAnalyzer.predict_sentiment(str(review))
                                sentiments.append(label)
                                confidences.append(confidence)
                            except:
                                sentiments.append("unknown")
                                confidences.append(0.0)
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        df['sentiment'] = sentiments
                        df['confidence'] = confidences
                        
                        # Display results
                        st.success("Analysis complete!")
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        sentiment_counts = df['sentiment'].value_counts()
                        
                        with col1:
                            st.metric("🟢 Positive", sentiment_counts.get('positive', 0))
                        with col2:
                            st.metric("🟡 Neutral", sentiment_counts.get('neutral', 0))
                        with col3:
                            st.metric("🔴 Negative", sentiment_counts.get('negative', 0))
                        
                        # Show dataframe
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Option 2: Manual input
    st.subheader("Option 2: Enter Multiple Reviews")
    manual_reviews = st.text_area(
        "Enter reviews (one per line):",
        height=200,
        placeholder="Review 1\nReview 2\nReview 3\n..."
    )
    
    if st.button("🔍 Analyze Manual Reviews"):
        if manual_reviews.strip():
            reviews_list = [r.strip() for r in manual_reviews.split('\n') if r.strip()]
            
            with st.spinner("Analyzing reviews..."):
                results = []
                for review in reviews_list:
                    try:
                        label, confidence = SentimentAnalyzer.predict_sentiment(review)
                        results.append({
                            'review': review,
                            'sentiment': label,
                            'confidence': confidence
                        })
                    except:
                        results.append({
                            'review': review,
                            'sentiment': 'unknown',
                            'confidence': 0.0
                        })
                
                df_results = pd.DataFrame(results)
                
                # Display results
                st.success(f"Analyzed {len(results)} reviews!")
                st.dataframe(df_results, use_container_width=True)
                
                # Download
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "manual_sentiment_results.csv",
                    "text/csv"
                )
        else:
            st.warning("Please enter at least one review.")

# Tab 3: Statistics
with tab3:
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Details")
        st.markdown("""
        - **Model:** cardiffnlp/twitter-roberta-base-sentiment-latest
        - **Fine-tuned on:** Food.com recipes dataset
        - **Task:** Multi-class sentiment classification
        - **Classes:** Positive, Neutral, Negative
        - **Framework:** Hugging Face Transformers
        """)
    
    with col2:
        st.subheader("🎯 Performance")
        st.markdown("""
        The model has been fine-tuned on thousands of food recipe reviews
        to accurately classify sentiments in culinary contexts.
        
        **Best for:**
        - Food reviews
        - Recipe feedback
        - Culinary experiences
        """)
    
    st.subheader("💡 Tips for Best Results")
    st.info("""
    - Write clear, detailed reviews
    - Include specific aspects (taste, texture, ease)
    - Use natural language
    - The model works best with English text
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit and Hugging Face Transformers</p>
        <p>MangeTaMain Project - Recipe Sentiment Analysis</p>
    </div>
""", unsafe_allow_html=True)
