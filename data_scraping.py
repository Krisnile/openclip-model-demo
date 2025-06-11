# clip_aggregator/data_scraping.py

import json
import time
import os # We'll use os.listdir and os.path.join

def fetch_reddit_posts(limit=50):
    """
    Simulates fetching posts, now by scanning a local_images directory
    and treating each image file as a post.
    """
    local_images_dir = "local_images" # Define your local images folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    print(f"Scanning '{local_images_dir}' for images...")
    
    found_posts = []
    # Ensure the directory exists
    if not os.path.exists(local_images_dir):
        print(f"Error: Directory '{local_images_dir}' not found. Please create it and add images.")
        return []

    # Iterate through files in the directory
    for filename in os.listdir(local_images_dir):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(local_images_dir, filename)
            # Create a 'title' from the filename for simplicity
            title = os.path.splitext(filename)[0].replace('_', ' ').title() 
            found_posts.append({"title": title, "url": file_path})

    print(f"Found {len(found_posts)} images in '{local_images_dir}'.")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        return found_posts[:limit]
    else:
        return found_posts # Return all if limit is 0 or None

if __name__ == "__main__":
    # Example usage for testing: fetch all found images
    posts = fetch_reddit_posts(limit=0) # Set limit=0 to get all images
    for post in posts:
        print(json.dumps(post, indent=2))