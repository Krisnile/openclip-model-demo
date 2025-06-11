# view_data.py
import sqlite3
import numpy as np
import json # For pretty printing

DATABASE_NAME = "clip_data.db"

def get_all_image_data_from_db():
    """Retrieves all image IDs, URLs, titles, and features from the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, url, features FROM images")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        _id, title, url, features_blob = row
        # Convert BLOB back to numpy array
        features = np.frombuffer(features_blob, dtype=np.float32) 
        results.append({
            "id": _id, 
            "title": title, 
            "url": url, 
            "features_shape": features.shape, # Just show shape, not full array
            "features_example": features[:5].tolist() # Show first 5 values as list
        })
    return results

if __name__ == "__main__":
    print("--- Fetching all stored data from the database ---")
    all_data = get_all_image_data_from_db()

    if all_data:
        print(f"Found {len(all_data)} entries in the database:")
        for item in all_data:
            # Using json.dumps for structured and readable output
            print(json.dumps(item, indent=2, ensure_ascii=False))
            print("-" * 30) # Separator for readability
    else:
        print("No data found in the database. Please run `python main.py` first to populate it.")
        