import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def load_data():
    data = pd.read_csv('movies.csv') 
    data['combined_features'] = data['title'] + ' ' + data['genres'] + ' ' + data['description'].fillna('')
    return data

def recommend_movies(movie_title, data, similarity_matrix):
    movie_title = movie_title.lower()
    if movie_title not in data['title'].str.lower().values:
        return ["Movie not found. Please try another title."]
    
    idx = data[data['title'].str.lower() == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended = [data.iloc[i[0]].title for i in sorted_scores]
    return recommended

def main():
    st.title("🎬 Movie Recommendation System")
    data = load_data()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    movie_name = st.text_input("Enter a movie title:")
    if st.button("Recommend"):
        recommendations = recommend_movies(movie_name, data, similarity_matrix)
        st.write("### Recommended Movies:")
        for movie in recommendations:
            st.write(movie)

if __name__ == '__main__':
    main()
