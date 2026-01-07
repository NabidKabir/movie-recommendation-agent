# movie-recommendation-agent

A recommendation agent that uses custom-built RAG database to tailor new movies based on individual taste

Here I have provided my own Letterboxd data to serve as a test for you and for personal use myself. However you can swap the files in /kb with your own Letterboxd CSV files by obtaining them from your account. You MUST have a letterboxd account in order for it to work, as the CSV ingestion is very specific. 

# Step 1: (Skip if merely want to test with my provided data)

Export your letterboxd data from their website. In the exported zip file you will find three files 
    ratings.csv
    watched.csv
    watchlist.csv
Swap these files with the provided files under the same name. This will allow your personal data to be used as the agent knowledge base

# Step 2

Make sure you have all the requirements as seen in requirements.txt set up. Run the file kb_ingest.py, this will create a folder known as rag_db within the directory, which will act as our knowledge base. 