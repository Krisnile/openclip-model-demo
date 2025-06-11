import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from clip_encoding import encode_text_with_clip
from data_storage import get_all_image_features

def search_images(query_text):
    """
    Performs a semantic search for images based on a natural language query.
    Returns Top N results based on cosine similarity.
    """
    print(f"User query: '{query_text}'")
    query_features = encode_text_with_clip(query_text)
    
    all_images_data = get_all_image_features()
    if not all_images_data:
        print("No images in the database to search.")
        return []

    image_features_list = [item["features"] for item in all_images_data]
    # Stack features into a 2D numpy array for efficient calculation
    image_features_matrix = np.array(image_features_list)

    # Calculate cosine similarity between query features and all image features
    # Reshape query_features for cosine_similarity to work correctly (1 sample, N features)
    similarities = cosine_similarity(query_features.reshape(1, -1), image_features_matrix).flatten()

    # Combine results with their similarities
    results = []
    for i, item in enumerate(all_images_data):
        results.append({
            "title": item["title"],
            "url": item["url"],
            "similarity": similarities[i]
        })

    # Sort by similarity in descending order
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:3] # Return Top 5 results

if __name__ == "__main__":
    # This block assumes the database has been populated by running main.py at least once.
    # To test independently, you'd need to manually populate the DB or ensure `main.py` has run.
    print("Running a sample search (ensure database is populated first).")
    search_results = search_images("beautiful beach photos")
    print("\nSearch Results:")
    if search_results:
        for result in search_results:
            print(f"Title: {result['title']}, Similarity: {result['similarity']:.2f}, URL: {result['url']}")
    else:
        print("No results found or database is empty.")