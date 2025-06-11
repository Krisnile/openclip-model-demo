import sqlite3
import numpy as np

DATABASE_NAME = "clip_data.db"

def init_db():
    """Initializes the SQLite database table."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            features BLOB
        )
    """)
    conn.commit()
    conn.close()

def clear_db():
    """Clears all data from the images table."""
    print(f"Clearing all data from '{DATABASE_NAME}'...")
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images")
    conn.commit()
    conn.close()
    print("Database cleared.")

def save_to_db(title, url, features):
    """Saves image data and CLIP features to the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    features_blob = features.tobytes() # Convert numpy array to bytes for BLOB storage
    cursor.execute("INSERT INTO images (title, url, features) VALUES (?, ?, ?)",
                   (title, url, features_blob))
    conn.commit()
    conn.close()

def get_all_image_features():
    """Retrieves all image IDs, URLs, titles, and features from the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, url, features FROM images")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        _id, title, url, features_blob = row
        # Ensure dtype matches how it was saved (e.g., np.float32)
        features = np.frombuffer(features_blob, dtype=np.float32)
        results.append({"id": _id, "title": title, "url": url, "features": features})
    return results

if __name__ == "__main__":
    init_db()
    # clear_db() # 测试清空
    print(f"Database '{DATABASE_NAME}' initialized.")
    
    # Example: Save a dummy entry
    dummy_features = np.random.rand(512).astype(np.float32) # Example 512-dim feature
    save_to_db(title="Test Image from DB Script", url="http://example.com/db_test.jpg", features=dummy_features)
    print("Dummy data saved to DB.")

    all_data = get_all_image_features()
    print("\nData in DB:")
    for item in all_data:
        print(f"ID: {item['id']}, Title: {item['title']}, URL: {item['url']}, Features Shape: {item['features'].shape}")