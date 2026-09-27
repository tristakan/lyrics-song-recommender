# Mood-Based Song Recommender from Billboard Lyrics (NLP)

Recommends a Billboard Hot 100 song that matches how you feel, based only on its **lyrics**. You enter a *happiness* score, a *cynical ↔ romantic* score and a year; the system finds the closest song and opens it on YouTube.

Built from **7,487 Billboard Hot 100 songs (2008–2023)**, with lyrics scraped, cleaned and filtered to **5,940 English songs**, then organized with TF-IDF, K-Means clustering, LDA topic modeling and sentiment analysis.

![t-SNE visualization of lyric clusters](images/tsne_clusters.png)

---

## How it works

```
Billboard Hot 100 (weekly, 2008–2023)
        │  BeautifulSoup
        ▼
Song list (7,487 unique songs) ──► Lyrics from AZLyrics
        │
        ▼
Cleaning + English filter (fastText language ID) ──► 5,940 songs
        │
        ▼
Tokenize → stop words → stem → lemmatize → TF-IDF
        │
        ├──► K-Means (k chosen by silhouette) ──► "cynical ↔ romantic" score (0–1)
        │
        └──► TextBlob sentiment ──► "happiness" score (−1 to 1)
        │
        ▼
User input (happiness, cynical↔romantic, year) ──► closest song ──► YouTube
```

## 1. Data collection: `webscrape.ipynb`, `webscrape2.ipynb`, `merging.ipynb`

- Scraped the **weekly Billboard Hot 100** chart pages with `requests` + `BeautifulSoup`, looping week by week (`timedelta(days=7)`) and keeping each unique *(song, artist)* pair.
- Scraped lyrics for every song from **AZLyrics**. Artist and song names had to be normalized to match the site's URLs: removing "Featuring", "X", "&", "+", punctuation, and special cases such as *The Weeknd → weeknd*.
- Songs whose lyrics weren't found were retried with a second scraper (`webscrape2.ipynb`) using extra name rules, and `merging.ipynb` filled the gaps with a **left join** on `song_name` + `artist_name`. About **1,430 songs** still couldn't be matched and were dropped.

| Stage | Songs |
|---|---|
| Unique Billboard songs (Jan 2008 – Mar 2023) | **7,487** |
| After lyric scraping, cleaning and English-only filter | **5,940** |

## 2. Cleaning and preprocessing: `lan_cleaning.ipynb`, `clustering.ipynb`

- Removed non-letters, line breaks and Excel artifacts (`_x000D_`).
- Detected language with **fastText** (`lid.176.bin`) and kept English lyrics only.
- Tokenized with NLTK, removed standard stop words **plus lyric filler** (*oh, yeah, la, ooh, uh, hey, chorus, verse, outro*), then applied **Porter stemming** and **WordNet lemmatization**.
- Normalized slang so variants count as one word: *wan/na → want*, *got → get*, *ya → you*.

![Most frequent words](images/word_frequency.png)

## 3. Clustering and topics: `clustering.ipynb`

- **TF-IDF** vectors (max 10,000 features, `min_df=10`, `max_df=0.5` to drop very rare and very common words).
- **Choosing k:** ran K-Means for k = 2…10 and picked the k with the highest **silhouette score**, which gave **k = 2**.
- Reduced to 100 dimensions with **Truncated SVD**, clustered with **K-Means**, and visualized with **t-SNE** (chart above).
- **LDA topic modeling** with 2 topics described the clusters:

| Cluster | Songs | Top words (LDA) | Interpretation |
|---|---|---|---|
| 0 | 1,302 | money, lil, aint, profanity | Hip-hop / street, "cynical" |
| 1 | 4,638 | baby, feel, time, girl, never | Love and relationships, "romantic" |

| Cluster 0 | Cluster 1 |
|---|---|
| ![Word cloud, cluster 0](images/wordcloud_cluster0.png) | ![Word cloud, cluster 1](images/wordcloud_cluster1.png) |

**From clusters to a score:** instead of a hard label, each song gets a **cynical ↔ romantic score from 0 to 1**, based on its distance to the two cluster centers (min-max scaled). Songs near the "street" center score close to 0, and songs near the "love" center score close to 1.

## 4. Sentiment: `classification.ipynb`

- **TextBlob polarity** of the full lyrics gives a *happiness* score from −1 (negative) to +1 (positive). Median = 0.05; most pop lyrics are close to neutral.
- The notebook also includes a **TF-IDF + cosine similarity** experiment for finding songs with similar lyrics.

## 5. Recommendation: `song_recommend.py`

```python
song_recommend(user_happy_rating=0.1, user_energetic_rating=0.1, year=2018)
```
1. Keep songs from the chosen **year**.
2. Search in a window around the user's scores, widening it step by step (±0.01, ±0.02, …) until songs are found.
3. Pick the song with the smallest **squared distance** to the user's (happiness, cynical↔romantic) point.
4. Open a **YouTube search** for that song.

Example from the notebook: the input mood returned **"ICU" by Coco Jones**.

## Project structure

```
├── webscrape.ipynb          # Billboard + lyrics scraping
├── webscrape2.ipynb         # retry for songs that failed
├── merging.ipynb            # merge lyric batches (left join)
├── lan_cleaning.ipynb       # cleaning + fastText English filter
├── clustering.ipynb         # preprocessing, TF-IDF, K-Means, LDA, t-SNE, scores
├── classification.ipynb     # sentiment + cosine-similarity experiment
├── song_recommend.py        # recommendation function
├── merged_dataframe.xlsx    # 7,487 songs with lyrics
├── en_lyrics.xlsx           # 5,940 English songs
└── en_lyrics_scale.csv      # final table with cluster, cluster_distance, sentiment
```

## How to run

```bash
git clone https://github.com/tristakan/lyrics-song-recommender.git
cd lyrics-song-recommender
pip install pandas numpy scikit-learn nltk textblob wordcloud openpyxl
python song_recommend.py
```
> Note: `song_recommend.py` reads `en_lyrics_scale.csv` from a hard-coded Windows path. Change it to `en_lyrics_scale.csv` before running.

## Limitations and next steps

- **Lyrics ≠ music.** Mood also comes from tempo and sound. Adding Spotify audio features (valence, energy) would improve recommendations.
- **TextBlob** was built for general text and misreads slang and sarcasm. A transformer sentiment model or LLM would handle lyrics better.
- **Only 2 clusters** were chosen by silhouette score, so the "cynical ↔ romantic" axis is coarse.
- In `song_recommend.py`, the second filter overwrites the first, so the happiness window isn't applied during filtering (the final distance still uses both scores). Fixing it is a one-line change.
- No user evaluation yet. A small survey (do people like the recommended song?) would measure quality.

## Team and credits

Group project for a text-mining course at **National Tsing Hua University** (2022–2023), originally hosted at [rxthlessbeats/Song_recommendation_by_lyrics](https://github.com/rxthlessbeats/Song_recommendation_by_lyrics).

**My contributions:** [e.g. lyric scraping and name-matching rules / preprocessing pipeline / clustering and the cynical–romantic score / recommendation function]

## Tech stack
Python · requests · BeautifulSoup · pandas · NLTK · fastText · scikit-learn (TF-IDF, K-Means, Truncated SVD, t-SNE, LDA) · TextBlob · WordCloud · matplotlib
