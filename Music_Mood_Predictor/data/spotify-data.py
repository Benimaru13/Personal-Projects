import requests
import base64
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
# === Step 1: Spotify API credentials ===

client_id = os.getenv("SPOTIFY_CLIENT_ID")  # Replace with your client ID or use environment variable
client_secret =  os.getenv("SPOTIFY_CLIENT_SECRET")  # Replace with your client secret or use environment variable

# === Step 2: Get access token ===
auth_str = f"{client_id}:{client_secret}"
b64_auth_str = base64.b64encode(auth_str.encode()).decode()

token_url = 'https://accounts.spotify.com/api/token'
headers = {
    'Authorization': f'Basic {b64_auth_str}',
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {'grant_type': 'client_credentials'}

res = requests.post(token_url, headers=headers, data=data)
access_token = res.json().get('access_token')

if not access_token:
    print("Failed to get token.")
    exit()

# === Step 3: Search for tracks ===
search_url = 'https://api.spotify.com/v1/search'
query = 'basketball'  # or any keyword/artist/genre you want
params = {
    'q': query,
    'type': 'track',
    'limit': 20  # You can adjust this
}
search_headers = {
    'Authorization': f'Bearer {access_token}'
}

response = requests.get(search_url, headers=search_headers, params=params)
tracks_json = response.json()

track_data = []
# === Step 4: Loop through tracks and get audio features ===
for item in tracks_json.get('tracks', {}).get('items', []):
    track_id = item['id']
    features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    features_res = requests.get(features_url, headers=search_headers)

    if features_res.status_code != 200:
        print(f"Skipping track {track_id} (no features found)")
        continue

    features = features_res.json()

    track_info = {
        'Track Name': item['name'],
        'Artist': item['artists'][0]['name'],
        'Album': item['album']['name'],
        'Release Date': item['album']['release_date'],
        'Spotify URL': item['external_urls']['spotify'],
        'acousticness': features['acousticness'],
        'danceability': features['danceability'],
        'energy': features['energy'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'loudness': features['loudness'],
        'speechiness': features['speechiness'],
        'tempo': features['tempo'],
        'valence': features['valence'],
        'key': features['key'],
        'mode': features['mode'],
        'time_signature': features['time_signature']
    }

    track_data.append(track_info)
    time.sleep(0.1)  # Pause to avoid rate limiting (optional)
    
    
# === Step 5: Save to CSV ===
df = pd.DataFrame(track_data)
df.to_csv('spotify_tracks.csv', index=False)

print("Data saved to spotify_tracks.csv")
