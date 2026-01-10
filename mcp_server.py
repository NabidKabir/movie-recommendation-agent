from fastmcp import FastMCP
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import tmdbsimple as tmdb
import asyncio
import functools
from typing import Callable, Any

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
        ratings[key] = row["Rating"]
    
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

    print(f"[TMDB]\tExtracting:\t{movie['title']} — Director: {movie.get('director', 'Unknown')}")


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


def normalize_metadata(movie: dict) -> dict:
    """
    Convert metadata into Chroma-safe scalar values.
    """
    safe = {}

    for k, v in movie.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, list):
            # Convert lists to comma-separated strings
            safe[k] = ", ".join(map(str, v))
        else:
            # Drop anything complex (dicts, objects)
            continue

    return safe

# Function to consiladte all knowledge base creation functions and actually set it up for use.
# This function is called in a sepearte .py file so as to not call everytime mcp server is ran. 
# As this process takes a while, is meant to be ingested every now and then when major updates or new csv files are added.
def kb_ingest():
    watched_docs, watchlist_docs, ratings_docs = load_csv_doc()
    print("CSV files loaded...")

    movies = merge_documents(watched_docs, watchlist_docs, ratings_docs)
    print(f"Documents merged with {len(movies)} items...")

    texts = []
    metadatas = []

    for movie in movies:
        tmdb_enriched = tmdb_ingestion_search(movie)
        texts.append(movie_to_text(tmdb_enriched))
        metadatas.append(normalize_metadata(tmdb_enriched))

    Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=chroma_path
    )

    print("Knowledge Base ready for use...")
    return {
        "status": "success",
        "movies_ingested": len(texts)
    }

# tmdbsimple library is a synchronous library, so in order to make asynchronous calls we must wrap function calls 
# in an async wrapper.
async def run_blocking(func, **kwargs):
    loop = asyncio.get_running_loop()
    partial = functools.partial(func, **kwargs)

    return await loop.run_in_executor(None, partial)

@movie_mcp.tool(title="Movie Knowledge Base Retrieval")
async def retrieve_personal_movies(query: str, top_k: int = 5) -> dict:
    """
        Function that retrieves relevant movie information from the knowledge base.

        Args:
            query: string that contains the user query to run a similarity search against knowledge base
            top_k: returns a dictionary of the top k elements returned

        Returns:
            dictionary with the query as well as any results in a list
    """

    if not os.path.exists(chroma_path):
        return {"status": "error", "message": "Chroma DB missing. Run kb_ingest() first."}
    
    vector_store = Chroma(
        collection_name="knowledge-base",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=chroma_path
    )

    # According to LLM response, this function is still blocking, however I cannot find any documentation citing that,
    # So I will keep it as using Chroma's built in function unless it breaks somewhere along the way.
    results = await vector_store.asimilarity_search("Love Exposure", k=top_k)
    output = []
    for doc in results:
        metadata = doc.metadata.copy()
        output.append({
            "title": metadata.get("title"),
            "year": metadata.get("year"),
            "watched": metadata.get("watched"),
            "watchlisted": metadata.get("watchlisted"),
            "rating": metadata.get("rating"),
            "director": metadata.get("director"),
            "cast": metadata.get("cast"),
            "genres": metadata.get("genres")
        })
    return {"status": "success", "results": output}

# watchlist documents held as a global so agent does not have to load CSV for every tmdb_movie_reccomend call
watchlist_docs = CSVLoader("./kb/watchlist.csv").load()

watchlist_keys = set()
for doc in watchlist_docs:
    row = parse_document(doc)
    key = (row["Name"].lower(), row["Year"])
    watchlist_keys.add(key)

@movie_mcp.tool(title="TMDB Movie Recommend")
async def tmdb_movie_recommend(*, similar_to: str | None = None,
                         genres: list[str] | None = None,
                         min_rating: float = 7,
                         top_k: int = 5):
    """
        Recommend movies using TMDB.
        Useful for when discovering new movies that are not on watchlist.
        Boosts movies that appear in the users watchlist

        Priority:
        1. Similar-to movie search
        2. Genre-based discovery fallback

        Args:
            similar_to: string string that contains data to use as a filter when searching for new movies
            genres: list of strings that contains the genres to look for when searching for new movies
            min_rating: float value that contains the minimum rating a movie has to have to be considered
            top_k: returns a dictionary of the top k elements returned
            
        Returns:
            dictionary with the query as well as any results in a list
    """

    results = []

    if similar_to:
        search = tmdb.Search()
        response = await run_blocking(search.movie, query=similar_to)

        if response.get("results"):
            movie_id = response["results"][0]["id"]
            similar = await run_blocking(tmdb.Movies(movie_id).similar_movies)
            results = similar.get("results", [])


    if not results and genres:
        discover_args = {
            "with_genres": ",".join(map(str, genres)),
            "sort_by": "popularity.desc"
        }
        if min_rating is not None:
            discover_args["vote_average_gte"] = min_rating

        discover = tmdb.Discover()
        response = await run_blocking(discover.movie, **discover_args)
        results = response.get("results", [])

    recommendations = []
    for movie in results[:top_k]:
        key = (movie["title"].lower(), movie.get("release_date", "")[:4])
        recommendations.append({
            "title": movie["title"],
            "year": movie.get("release_date", "")[:4],
            "rating": movie.get("vote_average"),
            "tmdb_id": movie["id"],
            "watchlist": key in watchlist_keys
        })

    if not recommendations:
        return {"status": "success", "results": [], "message": "No movies found matching the criteria."}

    return {"status": "success", "results": recommendations}

@movie_mcp.tool(title="Get Movie Cover")
async def get_movie_cover(tmdb_id: int | None = None, title: str | None = None, year: str | None = None):
    """
    Retrieve a movie poster image URL and TMDB page link.
    
    Args:
        tmdb_id: TMDB movie ID (preferred)
        title: Movie title if tmdb_id not provided
        year: Year to help locate movie by title
        
    Returns:
        dict with 'poster_url' and 'tmdb_url'
    """
    movie_info = None

    # If we don't have tmdb_id, search by title + year
    if tmdb_id is None and title:
        search = tmdb.Search()
        response = await run_blocking(search.movie, query=title)
        if response.get("results"):
            # If year is given, try to match
            if year:
                match = next((m for m in response["results"] if (m.get("release_date") or "")[:4] == year), None)
                movie_info = match or response["results"][0]
            else:
                movie_info = response["results"][0]
            tmdb_id = movie_info["id"]

    # If we now have tmdb_id, fetch full details
    if tmdb_id:
        details = tmdb.Movies(tmdb_id)
        info = await run_blocking(details.info)
        poster_path = info.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else None
        tmdb_url = f"https://www.themoviedb.org/movie/{tmdb_id}"
        return {"poster_url": poster_url, "tmdb_url": tmdb_url}
    
    return {"poster_url": None, "tmdb_url": None}

if __name__ == "__main__":
    movie_mcp.run(transport="http", host="0.0.0.0", port=8000)