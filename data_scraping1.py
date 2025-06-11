import json
import time

def fetch_reddit_posts(limit=50):
    """
    Simulates fetching posts with images from Reddit.
    In a real scenario, you'd use a Reddit API client (e.g., PRAW).
    """
    print(f"Fetching {limit} posts from Reddit (simulated)...")
    
    # Use real, publicly accessible image URLs for testing
    simulated_posts = [
        {"title": "Scenic Mountain Landscape", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/National_Park_of_Mongolia_%28cropped%29.jpg/1280px-National_Park_of_Mongolia_%28cropped%29.jpg"},
        {"title": "Cute Cat Photo", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Cute_Cat_at_Cat_Cafe.jpg/1280px-Cute_Cat_at_Cat_Cafe.jpg"},
        {"title": "Hong Kong Victoria Harbour at Night", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/HK_Victoria_Harbour.jpg/1280px-HK_Victoria_Harbour.jpg"},
        {"title": "Assortment of Fresh Fruits", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Various_fruits.jpg/1280px-Various_fruits.jpg"},
        {"title": "Colorful Abstract Painting", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Abstract_painting_%282015%29.jpg/1280px-Abstract_painting_%282015%29.jpg"},
        {"title": "Beautiful Sunset Beach", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Clouds_over_the_ocean%2C_Kauai%2C_Hawaii_%28Unsplash%29.jpg/1280px-Clouds_over_the_ocean%2C_Kauai%2C_Hawaii_%28Unsplash%29.jpg"},
        {"title": "Dense Forest Trail", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Forest_path_in_winter.jpg/1280px-Forest_path_in_winter.jpg"},
        {"title": "Desert Canyons", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Grand_Canyon_with_Thunderclouds.jpg/1280px-Grand_Canyon_with_Thunderclouds.jpg"},
        {"title": "Northern Lights", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Aurora_Borealis_in_Norway.jpg/1280px-Aurora_Borealis_in_Norway.jpg"},
        {"title": "Cozy Winter Cabin", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Log_cabin_in_the_winter_forest.jpg/1280px-Log_cabin_in_the_winter_forest.jpg"},
        {"title": "Iconic Golden Gate Bridge", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/GoldenGateBridge-001.jpg/1280px-GoldenGateBridge-001.jpg"},
        {"title": "Venice Gondola Ride", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Gondolas_in_Venice%2C_Italy.jpg/1280px-Gondolas_in_Venice%2C_Italy.jpg"},
        {"title": "Cherry Blossom Trees", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Cherry_blossoms_in_Tokyo.jpg/1280px-Cherry_blossoms_in_Tokyo.jpg"},
        {"title": "Eiffel Tower in Paris", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Eiffel_Tower_from_Champs_de_Mars%2C_Paris_May_2023.jpg/1280px-Eiffel_Tower_from_Champs_de_Mars%2C_Paris_May_2023.jpg"},
        {"title": "Great Wall of China", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/The_Great_Wall_of_China_at_Badaling.jpg/1280px-The_Great_Wall_of_China_at_Badaling.jpg"},
        {"title": "Majestic Taj Mahal", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Taj_Mahal_in_India.jpg/1280px-Taj_Mahal_in_India.jpg"},
        {"title": "Pyramids of Giza", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Pyramids_of_Giza_from_above.jpg/1280px-Pyramids_of_Giza_from_above.jpg"},
        {"title": "Ancient Machu Picchu", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Machu_Picchu_Peru.jpg/1280px-Machu_Picchu_Peru.jpg"},
        {"title": "Christ the Redeemer, Rio", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Christ_the_Redeemer_statue%2C_Rio_de_Janeiro.jpg/1280px-Christ_the_Redeemer_statue%2C_Rio_de_Janeiro.jpg"},
        {"title": "Sydney Opera House", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Sydney_Opera_House_Exterior_Detail.jpg/1280px-Sydney_Opera_House_Exterior_Detail.jpg"},
    ]
    time.sleep(0.1) # A small delay here is fine, but not as critical as after each request
    return simulated_posts[:limit]

if __name__ == "__main__":
    posts = fetch_reddit_posts(limit=5)
    for post in posts:
        print(json.dumps(post, indent=2))