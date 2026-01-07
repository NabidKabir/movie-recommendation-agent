# Movie Recommendation Agent

A personalized movie recommendation agent that uses a custom-built RAG database to suggest new movies based on individual taste.

This repository includes sample Letterboxd data for testing and personal use. You can replace the provided files in the `/kb` folder with your own Letterboxd CSV exports. A Letterboxd account is required to use your own data, as the CSV ingestion is tailored to their export format.

---

## Step 1: Prepare Your Letterboxd Data (Optional)

If you want to use your own data:

1. Export your Letterboxd data from your account.  
2. In the exported ZIP file, locate the following CSV files:  
   - `ratings.csv`  
   - `watched.csv`  
   - `watchlist.csv`  
3. Replace the corresponding files in the `/kb` folder with your own CSVs, keeping the same filenames.

> This allows the agent to use your personal movie history as the knowledge base.

---

## Step 2: Set Up the Knowledge Base

1. Install the dependencies listed in `requirements.txt`.  
2. Run `kb_ingest.py` to ingest your CSV data.  

This will create a folder called `rag_db` in the project directory, which serves as the agent’s knowledge base.

---

## Step 3: Running the MCP Server

The MCP server exposes your recommendation tools and handles agent queries.

```bash
python mcp_server.py
