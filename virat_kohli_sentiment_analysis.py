import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import nltk
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def scrape_tweets(query, limit=1000):
    """
    Scrape tweets using snscrape
    """
    try:
        import snscrape.modules.twitter as sntwitter
        
        tweets_list = []
        scraper = sntwitter.TwitterSearchScraper(query)
        
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break
            tweets_list.append({
                "Date": tweet.date,
                "Tweet": tweet.rawContent
            })
        
        print(f"Successfully scraped {len(tweets_list)} tweets")
        return pd.DataFrame(tweets_list)
    
    except Exception as e:
        print(f"Error scraping tweets: {e}")
        print("Falling back to CSV file...")
        return None

def load_csv_fallback():
    """
    Load tweets from CSV file as fallback
    """
    try:
        df = pd.read_csv("virat_kohli_tweets.csv")
        print(f"Loaded {len(df)} tweets from CSV file")
        return df
    except FileNotFoundError:
        print("CSV file not found. Creating sample data...")
        # Create sample data for demonstration
        sample_tweets = [
            "Virat Kohli is the best batsman in the world! #KingKohli",
            "Amazing performance by Virat Kohli today!",
            "Virat Kohli's century was incredible to watch",
            "Not impressed with Virat Kohli's recent form",
            "Virat Kohli needs to improve his batting technique",
            "What a player Virat Kohli is! Pure class!",
            "Virat Kohli's leadership skills are outstanding",
            "Disappointed with Virat Kohli's performance today",
            "Virat Kohli is a legend of the game",
            "Great to see Virat Kohli back in form"
        ]
        
        sample_dates = pd.date_range(start="2024-01-01", periods=len(sample_tweets), freq="D")
        
        df = pd.DataFrame({
            "Date": sample_dates,
            "Tweet": sample_tweets
        })
        
        # Save sample data
        df.to_csv("virat_kohli_tweets.csv", index=False)
        print("Created sample CSV file with 10 tweets")
        return df

def clean_tweet(tweet):
    """
    Clean tweet text by removing URLs, mentions, hashtags, digits, and punctuation
    """
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    tweet = re.sub(r"@\w+", "", tweet)
    
    # Remove hashtags but keep the text
    tweet = re.sub(r"#(\w+)", r"\1", tweet)
    
    # Remove digits
    tweet = re.sub(r"\d+", "", tweet)
    
    # Remove punctuation except apostrophes
    tweet = re.sub(r"[^\w\s\']", "", tweet)
    
    # Convert to lowercase
    tweet = tweet.lower().strip()
    
    # Remove extra whitespace
    tweet = re.sub(r"\s+", " ", tweet)
    
    return tweet

def classify_sentiment(polarity):
    """
    Classify sentiment based on polarity score
    """
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def perform_sentiment_analysis(df):
    """
    Perform sentiment analysis on tweets
    """
    # Clean tweets
    df["Cleaned_Tweet"] = df["Tweet"].apply(clean_tweet)
    
    # Perform sentiment analysis
    sentiments = []
    polarities = []
    
    for tweet in df["Cleaned_Tweet"]:
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        polarities.append(polarity)
        sentiments.append(classify_sentiment(polarity))
    
    df["Polarity"] = polarities
    df["Sentiment"] = sentiments
    
    return df

def plot_sentiment_distribution(df):
    """
    Plot bar chart of sentiment distribution
    """
    plt.figure(figsize=(10, 6))
    sentiment_counts = df["Sentiment"].value_counts()
    
    colors = ["#2E8B57", "#FF6B6B", "#FFD93D"]  # Green, Red, Yellow
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    
    plt.title("Sentiment Distribution of Tweets about Virat Kohli", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Tweets", fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, sentiment_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha="center", va="bottom", fontweight="bold")
    
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_wordcloud(df):
    """
    Generate wordcloud from all tweets
    """
    # Combine all cleaned tweets
    all_text = " ".join(df["Cleaned_Tweet"].dropna())
    
    # Create wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        max_words=100,
        colormap="viridis",
        contour_width=3,
        contour_color="steelblue"
    ).generate(all_text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud of Tweets about Virat Kohli", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the sentiment analysis pipeline
    """
    print("=" * 60)
    print("VIRAT KOHLI TWEET SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Step 1: Scrape tweets or load from CSV
    print("\n1. Fetching tweets...")
    query = "Virat Kohli lang:en"
    df = scrape_tweets(query, limit=1000)
    
    if df is None or len(df) == 0:
        df = load_csv_fallback()
    
    # Step 2: Perform sentiment analysis
    print("\n2. Performing sentiment analysis...")
    df = perform_sentiment_analysis(df)
    
    # Step 3: Display results
    print("\n3. Analysis Results:")
    print("-" * 40)
    
    # Display first few rows
    print("\nFirst 5 rows of the DataFrame:")
    print(df[["Date", "Tweet", "Cleaned_Tweet", "Sentiment"]].head())
    
    # Display sentiment counts
    print("\nSentiment Distribution:")
    sentiment_counts = df["Sentiment"].value_counts()
    print(sentiment_counts)
    
    # Display percentage distribution
    print("\nSentiment Distribution (%):")
    sentiment_percentages = (df["Sentiment"].value_counts(normalize=True) * 100).round(2)
    for sentiment, percentage in sentiment_percentages.items():
        print(f"{sentiment}: {percentage}%")
    
    # Step 4: Create visualizations
    print("\n4. Creating visualizations...")
    
    # Set style for better plots
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Plot sentiment distribution
    plot_sentiment_distribution(df)
    
    # Generate wordcloud
    generate_wordcloud(df)
    
    # Step 5: Save results
    print("\n5. Saving results...")
    output_file = "virat_kohli_sentiment.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Step 6: Summary statistics
    print("\n6. Summary Statistics:")
    print("-" * 40)
    print(f"Total tweets analyzed: {len(df)}")
    print(f"Average polarity score: {df['Polarity'].mean():.3f}")
    print(f"Most common sentiment: {df['Sentiment'].mode().iloc[0]}")
    
    # Display some example tweets for each sentiment
    print("\nExample tweets by sentiment:")
    for sentiment in ["Positive", "Neutral", "Negative"]:
        sentiment_tweets = df[df["Sentiment"] == sentiment]
        if len(sentiment_tweets) > 0:
            print(f"\n{sentiment} tweets:")
            for i, (_, row) in enumerate(sentiment_tweets.head(2).iterrows()):
                print(f"  {i+1}. {row['Tweet'][:80]}...")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
