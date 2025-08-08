
# 🧠 Twitter Sentiment Analysis: Virat Kohli Edition

This project analyzes public sentiment around **Virat Kohli**, one of India's most celebrated cricketers, by collecting tweets using `snscrape` and performing **Natural Language Processing (NLP)** using `TextBlob`.

It provides a visual and statistical breakdown of how people feel about Virat Kohli on Twitter.

---

## 📌 Features

- ✅ Live tweet scraping using `snscrape`
- 🧹 Tweet cleaning and preprocessing
- 📊 Sentiment classification (Positive / Neutral / Negative)
- 📈 Visualization (bar chart + word cloud)
- 💾 Save results to CSV
- 📋 Summary statistics & example tweets

---

## 🖥️ How the Project Works

1. **Scrape Tweets**  
   - Uses `snscrape` to collect tweets based on a query (`Virat Kohli lang:en`).
   - If scraping fails, it loads data from a CSV or creates 10 sample tweets.

2. **Clean Tweets**  
   - Removes URLs, mentions, hashtags, digits, and punctuation.
   - Converts text to lowercase and removes extra whitespace.

3. **Perform Sentiment Analysis**  
   - Uses `TextBlob` to calculate polarity scores.
   - Classifies sentiment as:
     - **Positive** if polarity > 0.1
     - **Negative** if polarity < -0.1
     - **Neutral** otherwise

4. **Visualize Results**  
   - Bar chart showing sentiment distribution.
   - Word cloud showing most frequent words.

5. **Output**  
   - Saves a CSV file with all tweets and their sentiment analysis.
   - Prints a summary with total tweets, average polarity, and example tweets.

---

## ⚙️ Installation

### 🔹 Install Python Libraries

Run the following command in your terminal or command prompt:

```bash
pip install pandas numpy matplotlib seaborn textblob nltk wordcloud snscrape
```

### 🔹 NLTK Setup (one-time only)

The script auto-downloads required NLTK data (`punkt`) if it's missing.

---

## 🚀 How to Run the Project

1. **Download or Clone the Repository**

```bash
git clone https://github.com/your-username/virat-kohli-sentiment-analysis.git
cd virat-kohli-sentiment-analysis
```

2. **Run the Python Script**

```bash
python virat_kohli_sentiment_analysis.py
```

3. **View the Results**

- Sentiment bar chart and word cloud will appear.
- Output saved as `virat_kohli_sentiment.csv`.

---

## 📊 Sample Output

- **CSV File** → `virat_kohli_sentiment.csv` containing:
  - Date
  - Original tweet
  - Cleaned tweet
  - Sentiment polarity
  - Sentiment category

- **Visualizations:**
  - Sentiment distribution (bar chart)
  - Word cloud of most frequent terms

- **Console Output:**
  - Number of tweets scraped
  - Sentiment percentages
  - Example tweets per sentiment

---

## 🗂️ Project Structure

```
virat-kohli-sentiment-analysis/
│
├── virat_kohli_sentiment_analysis.py      # Main Python script
├── virat_kohli_tweets.csv                 # (Optional) fallback tweet data
├── virat_kohli_sentiment.csv              # Output after analysis
└── README.md                              # Project documentation
```

---

## 📎 Example Output Preview

**Sentiment Distribution**
```
Positive: 62%
Neutral: 24%
Negative: 14%
```

**Sample Tweet (Positive)**
> "Virat Kohli is the best batsman in the world! #KingKohli"

---
 **Created by Pranav Patil**
