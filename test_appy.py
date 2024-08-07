import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Add custom CSS for background image
background_image_url = "'https://image.tmdb.org/t/p/original'"  # Replace with your image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url({background_image_url});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# st.set_page_config(layout='wide')

url = 'https://image.tmdb.org/t/p/original'
link = link = "https://raw.githubusercontent.com/Lovelylove03/movie-testing-/main/df_ml%20-%20df_ml.csv"
# Rest of your code follows

df= pd.read_csv(link)
st.title('Système de recommandations de films pour cinema')
st.divider()
# Display DataFrame for debugging
st.write("DataFrame Head:")
st.write(df.head())

st.write("DataFrame Columns:")
st.write(df.columns)
with st.expander('Data'):
    st.write('**Raw data**')
col1, col2, col3= st.columns(3)

with st.sidebar:
    st.header("Choisir un film")
    
    time = st.radio('Années',["All","2020's","2010's","2000's","90's","80's","70's","60's","50's","40's","30's","20's"])

    if time == "2020's":
        df = df.loc[df['startYear'].astype(str).str.startswith('202')]
    elif time == "2010's":
        df = df.loc[df['startYear'].astype(str).str.startswith('201')]
    elif time == "2000's":
        df = df.loc[df['startYear'].astype(str).str.startswith('200')]
    elif time == "90's":
        df = df.loc[df['startYear'].astype(str).str.startswith('199')]
    elif time == "80's":
        df = df.loc[df['startYear'].astype(str).str.startswith('198')]
    elif time == "70's":
        df = df.loc[df['startYear'].astype(str).str.startswith('197')]
    elif time == "60's":
        df = df.loc[df['startYear'].astype(str).str.startswith('196')]
    elif time == "50's":
        df = df.loc[df['startYear'].astype(str).str.startswith('195')]
    elif time == "40's":
        df = df.loc[df['startYear'].astype(str).str.startswith('194')]
    elif time == "30's":
        df = df.loc[df['startYear'].astype(str).str.startswith('193')]
    elif time == "20's":
        df = df.loc[df['startYear'].astype(str).str.startswith('192')]

    st.write("Veuillez sélectionner un film pour lequel nous vous indiquerons  plusieurs équivalences à visionner!")
    with st.form("filters"):
        movie = st.selectbox("films", df["primaryTitle"])
        submitted = st.form_submit_button("Submit")

X = df.drop(columns=['tconst','primaryTitle','poster_path'])
X_scaler = X
model = NearestNeighbors(n_neighbors=4, algorithm='brute').fit(X)


if submitted:

    X_index = df.loc[df['primaryTitle'].str.contains(movie)].index
    a = model.kneighbors(X_scaler.loc[X_index], return_distance=False)
    df1 = df.loc[df['primaryTitle'].str.contains(movie, case = False)]

    if len(df1) > 1 :
        for x in range (len(df1)):
            col2.header(df1.iloc[x,:]['primaryTitle'])
            col2.subheader (df1.iloc[x,:]['startYear'])
            col2.image(url + df1.iloc[x,:]['poster_path'],use_column_width='auto')
    else:




        col1.header(df.iloc[a[0][1]]['primaryTitle'])
        col1.subheader (df.iloc[a[0][1]]['startYear'])
        col1.image(url + df.iloc[a[0][1]]['poster_path'],use_column_width='auto')

        col2.header(df.iloc[a[0][2]]['primaryTitle'])
        col2.subheader (df.iloc[a[0][2]]['startYear'])
        col2.image(url + df.iloc[a[0][2]]['poster_path'],use_column_width='auto')

        col3.header(df.iloc[a[0][3]]['primaryTitle'])
        col3.subheader (df.iloc[a[0][3]]['startYear'])
        col3.image(url + df.iloc[a[0][3]]['poster_path'],use_column_width='auto')   




# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(X, X)

# Integrate with Streamlit
st.write("Cosine Similarity Matrix:")
st.write(similarity_matrix)
import matplotlib.pyplot as plt
# Univariate Analysis: Pie charts for categorical variables
st.write("Univariate Analysis: Pie Charts")
for column in df.select_dtypes(include=['object']).columns:
    if df[column].nunique() < 10:  # Limit to categorical columns with fewer unique values
        fig, ax = plt.subplots()
        df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_title(f'Pie Chart of {column}')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

import numpy as np

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Action", "Animation", "Biography","Comedy", "Crime", "Drama", "Family", "Fantasy" "Adventure", "Mystery", "Romance", "Sci-Fi","Science Fiction","Sport","Thriller","War","Western"])

st.area_chart(chart_data)
