import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Prompt the user to enter text
text = input("Enter the text you want to analyze (English Only):\n")

# Text preprocessing using spaCy
doc = nlp(text)

# Extract features (e.g., tokenization, lemmatization)
tokens = [token.lemma_ for token in doc]

# Join the tokens back into a processed text
processed_text = " ".join(tokens)

# Sentiment analysis using vaderSentiment
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = analyzer.polarity_scores(processed_text)

# Categorize sentiment
if sentiment_scores['compound'] >= 0.05:
    sentiment_category = 'Positive'
elif sentiment_scores['compound'] <= -0.05:
    sentiment_category = 'Negative'
else:
    sentiment_category = 'Neutral'

# Create a DataFrame for sentiment scores
sentiment_df = pd.DataFrame(sentiment_scores.items(), columns=['Category', 'Score'])
sentiment_df['Category'] = sentiment_df['Category'].replace({'pos': 'Positive', 'neg': 'Negative', 'neu': 'Neutral', 'compound': 'Compound'})

# Display results
print(f"\nOriginal text: {text}")
print(f"Processed text: {processed_text}")
print(f"Predicted Sentiment: {sentiment_category}")
print("\nSentiment Scores:")
print("-----------------")
print(sentiment_df.to_string(index=False))
