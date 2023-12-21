import tkinter as tk
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

if len(sys.argv) != 2:
        sys.exit("Usage: python ALP.py moviedataset.csv")

df = pd.read_csv('moviedataset.csv')

missing_value = df.isna()

df['year']=df['year'].str.replace("-","")

romawi = ["(",")","I","V"," ","X"]
for a in romawi:
  df['year']=df['year'].str.replace(a,"")

df.dropna(subset=['Overview'], inplace=True)
df['year'].fillna(df['year'].mode()[0], inplace=True)

duplicate_movies = df[df.duplicated(subset='movie title', keep=False)]

df=df.drop_duplicates(subset=['movie title','year'],keep='last')

df['Run Time'] = df['Run Time'].str.replace(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?', '0', regex=True).str.replace("(estimated)", '').str.replace("(", '').str.replace(")", '').str.replace("not-released", '0')

df['Generes']=df['Generes'].str.strip('[]').str.replace(' ','').str.replace("'",'')
df['Generes']=df['Generes'].str.split(',')

df['Plot Kyeword']=df['Plot Kyeword'].str.strip('[]').str.replace(' ','').str.replace("'",'')
df['Plot Kyeword']=df['Plot Kyeword'].str.split(',')

df['Top 5 Casts']=df['Top 5 Casts'].str.strip('[]').str.replace("'",'')
df['Top 5 Casts']=df['Top 5 Casts'].str.split(',')

df['User Rating']=df['User Rating'].str.replace("K",'000').str.replace("M", '000000')

df['User Rating'] = pd.to_numeric(df['User Rating'])
normalize_user_rating = (df['User Rating']-df['User Rating'].min()) / (df['User Rating'].max()-df['User Rating'].min())

df = df[df['Rating'] != 'no-rating']

df['Rating'] = pd.to_numeric(df['Rating'])
df['Modified_Rating'] =  normalize_user_rating + df['Rating']

df = df[['movie title', 'year', 'Generes', 'Run Time', 'Overview', 'Top 5 Casts', 'Director', 'Rating', 'Modified_Rating', 'Writer', 'Plot Kyeword']]

def list_to_string(genre_list):
    return ', '.join(genre_list)

df['Generes'] = df['Generes'].apply(list_to_string)

def list_to_string(keyword_list):
    return ', '.join(keyword_list)

df['Plot Kyeword'] = df['Plot Kyeword'].apply(list_to_string)

def list_to_string(cast):
    return ', '.join(cast)

df['Top 5 Casts'] = df['Top 5 Casts'].apply(list_to_string)

top_Genres = df.groupby('Generes')[['Rating']].mean().sort_values('Rating',ascending=False).head(10).round(2)
top_Genres.reset_index(inplace=True)

df['tags'] = df['Overview'] + df['Generes'] + df['Plot Kyeword'] + df['Top 5 Casts'] + df['Writer']
df['movie_id'] = range(1, len(df) + 1)

new_df = df[['movie_id', 'movie title', 'tags']]

new_df.rename(columns={'movie title': 'movie_title'}, inplace=True)
new_df['movie_title'] = new_df['movie_title'].str.lower()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(new_df['tags'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cos_df = pd.DataFrame(cosine_sim, columns=new_df.movie_title)
cos_df['Series_Title'] = new_df.movie_title
cos_df = cos_df[['Series_Title'] + cos_df.columns[:-1].to_list()]
cos_df.head()

idxs = pd.Series(new_df.index, index=new_df.movie_title)


import tkinter as tk
from tkinter import ttk

entry = None
result_text = None

def get_recommendations(title):
    # Fungsi rekomendasi yang sama seperti yang Anda berikan sebelumnya
    idx = idxs[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

def show_recommendations():
    global entry, result_text
    title = entry.get().lower()

    if title not in idxs:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Movie not found. Please enter a valid movie title.")
    else:
        recommendations = get_recommendations(title)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, recommendations[['movie title', 'Rating']].to_string(index=False))

# Fungsi untuk menampilkan UI
def main():
    global entry, result_text
    root = tk.Tk()
    root.title("MovAI")

    label = tk.Label(root, text="Enter Your Favorite Movie:")
    label.pack(pady=10)

    entry = ttk.Entry(root, width=30)
    entry.pack(pady=10)

    button = ttk.Button(root, text="Get Recommendations", command=show_recommendations)
    button.pack(pady=10)

    result_text = tk.Text(root, height=10, width=50)
    result_text.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()