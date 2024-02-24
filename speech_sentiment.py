import speech_recognition as sr
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using NLTK's SentimentIntensityAnalyzer.
    Returns the sentiment label and sentiment scores.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment_label = "Positive"
    elif compound_score <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return sentiment_label, sentiment_scores

def speech_sentiment_analyzer():
    """
    Listens to the user's speech and analyzes its sentiment.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("\nSpeak something (English Only)...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        print("\nYou said:", text)

        # Analyze sentiment
        sentiment_label, sentiment_scores = analyze_sentiment(text)
        print(f"Predicted Sentiment: {sentiment_label}")

        # Create a DataFrame from sentiment scores
        sentiment_df = pd.DataFrame({
            'Category': ['Negative', 'Neutral', 'Positive', 'Compound'],
            'Score': [sentiment_scores['neg'], sentiment_scores['neu'], 
                      sentiment_scores['pos'], sentiment_scores['compound']]
        })

        # Print sentiment scores
        print("\nSentiment Scores:")
        print("-----------------")
        print(sentiment_df.to_string(index=False))

    except sr.UnknownValueError:
        print("Sorry, could not understand your speech.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    speech_sentiment_analyzer()
