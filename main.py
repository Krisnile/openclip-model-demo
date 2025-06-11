import os
import time
import subprocess # Use subprocess for system commands more safely
import uvicorn

# Import functions from respective modules
from data_scraping import fetch_reddit_posts
from clip_encoding import load_image, encode_image_with_clip
from data_storage import init_db, clear_db, save_to_db, DATABASE_NAME
# user_query and app are not directly called here, but their underlying functions are used
# and app.py is run as a separate process by uvicorn.

def run_crawler_and_populate_db(limit=50):
    """Runs the data fetching and CLIP encoding, then populates the database."""
    print("--- Starting Data Scraping and CLIP Encoding ---")
    
    # Initialize the database
    init_db()
    print(f"Database '{DATABASE_NAME}' initialized/checked.")

    clear_db() 

    # Step 1: Data Scraping
    fetch_start_time = time.time()
    posts = fetch_reddit_posts(limit=limit)
    fetch_end_time = time.time()
    print(f"Data fetching completed in {fetch_end_time - fetch_start_time:.2f} seconds.")
    print(f"Fetched {len(posts)} posts.")

    # Step 2 & 3: CLIP Encoding and Storing Data
    encoding_start_time = time.time()
    processed_count = 0
    for i, post in enumerate(posts):
        print(f"Processing post {i+1}/{len(posts)}: {post['title']}")
        image = load_image(post["url"])
        if image:
            features = encode_image_with_clip(image)
            if features is not None:
                save_to_db(post["title"], post["url"], features)
                processed_count += 1
            else:
                print(f"  Skipping post '{post['title']}' due to encoding failure (might be invalid image).")
        else:
            print(f"  Skipping post '{post['title']}' due to image download failure.")
    encoding_end_time = time.time()
    print(f"CLIP encoding and data storage completed in {encoding_end_time - encoding_start_time:.2f} seconds.")
    print(f"Successfully processed and stored {processed_count} images.")
    print("--- Data Population Complete ---")

if __name__ == "__main__":
    # Create necessary directory for templates if it doesn't exist
    template_dir = "templates"
    index_html_path = os.path.join(template_dir, "index.html")

    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print(f"Created directory: {template_dir}")
    
    # Create index.html if it doesn't exist (or overwrite for fresh start)
    # For simplicity, we'll write the content directly. In a real scenario, you might
    # prefer to have this file pre-created by the user or copy from a source.
    # The content of index.html is provided in the previous section.
    # You would typically place the HTML content directly here.
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLIP Image Search</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }
        .container { max-width: 960px; margin: auto; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .search-form { display: flex; margin-bottom: 40px; justify-content: center; }
        .search-form input[type="text"] { 
            flex-grow: 1; 
            padding: 12px 15px; 
            border: 1px solid #ced4da; 
            border-radius: 6px; 
            font-size: 1.1em;
            max-width: 500px; /* Limit input width */
        }
        .search-form button { 
            padding: 12px 25px; 
            background-color: #007bff; 
            color: white; 
            border: none; 
            border-radius: 6px; 
            cursor: pointer; 
            margin-left: 10px; 
            font-size: 1.1em;
            transition: background-color 0.3s ease;
        }
        .search-form button:hover {
            background-color: #0056b3;
        }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 25px; }
        .results { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 25px; 
            padding: 10px;
        }
        .result-item { 
            border: 1px solid #e9ecef; 
            padding: 18px; 
            border-radius: 10px; 
            text-align: center; 
            background-color: #fcfcfc;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .result-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        }
        .result-item img { 
            max-width: 100%; 
            height: 200px; 
            object-fit: cover; 
            border-radius: 6px; 
            margin-bottom: 15px; 
            display: block; /* Ensures image takes full width of its container */
            margin-left: auto;
            margin-right: auto;
        }
        .result-item h3 { 
            margin-top: 0; 
            font-size: 1.2em; 
            color: #34495e; 
            margin-bottom: 8px;
        }
        .result-item p { 
            font-size: 0.95em; 
            color: #7f8c8d; 
            margin-bottom: 0;
        }
        .no-results {
            text-align: center;
            font-size: 1.2em;
            color: #6c757d;
            padding: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CLIP Image Search</h1>
        <form class="search-form" action="/search" method="get">
            <input type="text" name="query" placeholder="Enter your search query..." value="{{ query if query else '' }}">
            <button type="submit">Search</button>
        </form>

        {% if results %}
            <h2>Search Results for "{{ query }}"</h2>
            <div class="results">
                {% for result in results %}
                    <div class="result-item">
                        <img src="{{ result.url }}" alt="{{ result.title }}">
                        <h3>{{ result.title }}</h3>
                        <p>Similarity: {{ "%.2f" % (result.similarity * 100) }}%</p>
                    </div>
                {% endfor %}
            </div>
        {% elif query %}
            <p class="no-results">No results found for "{{ query }}".</p>
        {% endif %}
    </div>
</body>
</html>
    """
    if not os.path.exists(index_html_path):
        with open(index_html_path, "w") as f:
            f.write(html_content)
        print(f"Created '{index_html_path}'.")

    # Install dependencies
    print("--- Installing Dependencies (if not already met) ---")
    try:
        # Use subprocess.check_call to ensure the command runs successfully
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("--- Dependencies Installation Complete ---")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("Please ensure pip is installed and accessible.")
        exit(1) # Exit if dependencies cannot be installed

    # Run the crawler and populate the database
    run_crawler_and_populate_db()

    # Start the web service
    print("\n--- Starting Web Service (FastAPI) ---")
    print("Access the web service at: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the service.")
    
    # Run the FastAPI app using uvicorn.run. This blocks until the server is stopped.
    # It points to 'app:app' meaning the 'app' object in 'app.py'
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)