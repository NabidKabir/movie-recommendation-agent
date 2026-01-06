from fastmcp import FastMCP
import requests
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import httpx
import asyncio
import csv
import tmdbsimple as tmdb

# Load env files to access TDMB API KEY
load_dotenv()

# Using tmdbsimple library for easy access to api functions, loading api key with module
tmdb.API_KEY = os.environ["TMDB_API_KEY"]

# Creating FastMCP instance on our MCP Server
movie_mcp = FastMCP("Movie_Server")

chroma_path = "rag_db"

# Function to load all of our CSV data regarding our personal knowledge base
# Contained within a function to allow for ingestion at function call instead of runtime
# Theoretically useful incase instead of static files for kb, wanted to implement a continuously growing database
def load_csv_doc(watched_path="./kb/watched.csv", watchlist_path="./kb/watchlist.csv", ratings_path="./kb/ratings.csv"):
    watched_docs = CSVLoader(watched_path).load()
    watchlist_docs = CSVLoader(watchlist_path).load()
    ratings_docs = CSVLoader(ratings_path).load()

    return watched_docs, watchlist_docs, ratings_docs

# Function to parse document into a dictionary for access
def parse_document(doc):
    data = {}
    for line in doc.page_content.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data

# Function to  merge documents and consolidate into one dictionary for embedding access
def merge_documents(watched_docs, watchlist_docs, ratings_docs):
    watched = {}
    watchlist = {}
    ratings = {}

    for doc in watched_docs:
        row = parse_document(doc)
        key = (row["Name"], row["Year"])
        watched[key] = {
            "title": row["Name"],
            "year": row["Year"],
            "watched": True,
            "watchlisted": False,
            "watched_date": row.get("Date"),
            "rating": None,
        }

    for doc in ratings_docs:
        row = parse_document(doc)
        key = (row["Name"], row["Year"])
        ratings[key] = {
            "rating": row["Rating"]
        }
    
    for doc in watchlist_docs:
        row = parse_document(doc)
        key = (row["Name"], row["Year"])
        watchlist[key] = {
            "title": row["Name"],
            "year": row["Year"],
            "watched": False,
            "watchlisted": True,
            "watched_date": row.get("Date"),
            "rating": None,
        }
    
    merged = {}

    for key, movie in watched.items():
        rating = ratings.get(key)
        if rating is not None and rating != "":
            try:
                movie["rating"] = float(rating)
            except ValueError:
                movie["rating"] = None
        else:
            movie["rating"] = None
        merged[key] = movie
    
    for key, movie in watchlist.items():
        if key not in merged:
            merged[key] = movie
    
    return list(merged.values())


# Function to enrich embeddings with more data gathered from tmdb search
# CSV data only contained names, years, ratings of movies. Need data on genre, director, etc. to
#  perform an accurate similarity search.
def tmdb_ingestion_search(movie):
    search = tmdb.Search()
    response = search.movie(
        query=movie["title"],
        year=movie["year"]
    )

    if not response.get("results"):
        return movie
    
    best = response["results"][0]
    tmdb_id = best["id"]
    movie["tmdb_id"] = tmdb_id

    details = tmdb.Movies(tmdb_id)
    info = details.info()
    credit = details.credits()
    keywords = details.keywords()

    movie["genres"] = [genre["name"] for genre in info.get("genres", [])]
    movie["director"] = next(
        (cast["name"] for cast in credit.get("crew", [])
         if cast.get("job") == "Director"),
         None
    )
    movie["cast"] = [cast["name"] for cast in credit.get("cast", [])[:5]]
    movie["keywords"] = [keyword["name"] for keyword in keywords.get("keywords", [])]

    return movie

# Function to convert movie dictionary into embeddable text for chroma db
def movie_to_text(movie):
    parts = [f"Title: {movie['title']} ({movie['year']})."]

    if movie.get("genres"):
        parts.append(f"Genres: {', '.join(movie['genres'])}.")
    if movie.get("director"):
        parts.append(f"Directed by {movie['director']}.")
    if movie.get("cast"):
        parts.append(f"Starring {', '.join(movie['cast'])}.")
    if movie.get("keywords"):
        parts.append(f"Themes: {', '.join(movie['keywords'][:5])}.")

    if movie["watched"]:
        parts.append("The user has watched this movie.")
        if movie.get("rating"):
            parts.append(f"The user rated it {movie['rating']} stars.")
    else:
        parts.append("This movie is on the user's watchlist.")

    return " ".join(parts)

def kb_ingest():
    watched_docs, watchlist_docs, ratings_docs = load_csv_doc()

    movies = merge_documents(watched_docs, watchlist_docs, ratings_docs)

    texts = []
    metadatas = []

    for movie in movies:
        tmdb_enriched = tmdb_ingestion_search(movie)
        texts.append(movie_to_text(tmdb_enriched))
        metadatas.append(tmdb_enriched)

    Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=chroma_path
    ).persist()

    return {
        "status": "success",
        "movies_ingested": len(texts)
    }

