from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import time

from user_query import search_images # Import the search function

app = FastAPI()

app.mount("/local_images", StaticFiles(directory="local_images"), name="static_images")

# Configure Jinja2Templates to look for templates in the 'templates' directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the initial search page."""
    return templates.TemplateResponse("index.html", {"request": request, "query": "", "results": []})

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = None):
    """Handles search queries and displays results."""
    results = []
    if query:
        search_start_time = time.time()
        results = search_images(query)
        search_end_time = time.time()
        print(f"Search response time for '{query}': {search_end_time - search_start_time:.2f} seconds")
    
    return templates.TemplateResponse("index.html", {"request": request, "query": query, "results": results})

# This part is for running the app directly if this file is the entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)