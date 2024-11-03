from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load environment variables from .env file
load_dotenv()

# Spotify API Authentication
def authenticate_spotify():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope="user-read-recently-played user-library-read playlist-read-private"
    ))

# Data Collection
def fetch_listening_history(sp):
    results = sp.current_user_recently_played(limit=50)
    track_ids = [item['track']['id'] for item in results['items']]
    track_names = [item['track']['name'] for item in results['items']]
    timestamps = [item['played_at'] for item in results['items']]
    return track_ids, track_names, timestamps

# Retrieve Audio Features
def retrieve_audio_features(sp, track_ids):
    features = sp.audio_features(tracks=track_ids)
    return pd.DataFrame(features)

# Data Preprocessing
def preprocess_data(features_df, track_names, timestamps):
    # Organize data into a DataFrame
    features_df['timestamp'] = timestamps
    features_df['track_name'] = track_names

    # Handle missing values
    features_df = features_df.dropna()

    # Time Aggregation
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    features_df.set_index('timestamp', inplace=True)
    # Only aggregate numeric columns
    daily_data = features_df.select_dtypes(include=[np.number]).resample('D').mean()

    # Mood Labeling
    daily_data['mood_label'] = daily_data.apply(lambda row: classify_mood(row), axis=1)

    # Feature Engineering
    daily_data['energy_median'] = features_df['energy'].resample('D').median()
    daily_data['tempo_sum'] = features_df['tempo'].resample('D').sum()

    return features_df, daily_data

# Model Building and Evaluation
def build_and_evaluate_model(daily_data):
    X = daily_data[['energy', 'tempo', 'danceability', 'valence', 'energy_median', 'tempo_sum']]  # Features
    y = daily_data['mood_label']  # Labels

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if we have enough data for GridSearchCV
    if len(X_train) > 1:
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        # Fall back to default RandomForest model if not enough data
        best_model = RandomForestClassifier(random_state=42)
        best_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return best_model

# User Interaction Options
def user_options(daily_data, data, sp):
    while True:
        print("\nSelect an option:")
        print("1. See overall mood trend")
        print("2. Check your current mood")
        print("3. Look at your mood based on songs you listened to last week")
        print("4. See your top played songs in the last week and their mood")
        print("5. Analyze the mood of a playlist")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            visualize_mood_trend(daily_data)
        elif choice == '2':
            check_current_mood(daily_data)
        elif choice == '3':
            analyze_last_week_mood(daily_data)
        elif choice == '4':
            top_played_songs_last_week(data)
        elif choice == '5':
            analyze_playlist_mood(sp)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

# Visualization Functions
def visualize_mood_trend(daily_data):
    plt.figure(figsize=(10, 6))
    plt.plot(daily_data.index, daily_data['valence'], marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Valence (Mood Proxy)')
    plt.title('Mood Trend Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def check_current_mood(daily_data):
    latest_data = daily_data.iloc[-1]
    if latest_data['valence'] > 0.5:
        print("Your current mood is: Happy")
    else:
        print("Your current mood is: Sad")

def analyze_last_week_mood(daily_data):
    last_week_data = daily_data.loc[daily_data.index >= (daily_data.index.max() - pd.Timedelta(days=7))]
    avg_valence = last_week_data['valence'].mean()
    if avg_valence > 0.5:
        print("Your mood over the last week was mostly positive.")
    else:
        print("Your mood over the last week was mostly negative.")

    # Provide context on why the mood was this way
    context = last_week_data[['danceability', 'energy', 'valence', 'tempo']].mean()
    print("\nMood context for the last week:")
    print(f"Average Danceability: {context['danceability']:.2f}")
    print(f"Average Energy: {context['energy']:.2f}")
    print(f"Average Valence: {context['valence']:.2f}")
    print(f"Average Tempo: {context['tempo']:.2f}")

    # Save context to CSV
    context_df = pd.DataFrame([context])
    context_df.to_csv('last_week_mood_context.csv', index=False)

def top_played_songs_last_week(data):
    last_week_tracks = data.loc[data.index >= (data.index.max() - pd.Timedelta(days=7))]
    top_tracks = last_week_tracks['track_name'].value_counts().head(10)
    top_tracks_data = last_week_tracks[last_week_tracks['track_name'].isin(top_tracks.index)]
    top_tracks_data = top_tracks_data.copy()
    top_tracks_data['mood_label'] = top_tracks_data.apply(lambda row: classify_mood(row), axis=1)

    print("\nTop Played Songs in Last Week and Their Mood:")
    for track_name, count in top_tracks.items():
        mood = top_tracks_data[top_tracks_data['track_name'] == track_name]['mood_label'].iloc[0]
        print(f"Track Name: {track_name} - Mood: {mood} - Play Count: {count}")

    # Save top played songs to CSV
    top_tracks_data[['track_name', 'mood_label']].to_csv('top_played_songs_last_week.csv', index=False)

def analyze_playlist_mood(sp):
    playlist_name = input("Enter the name of the playlist you want to analyze: ")
    playlists = sp.current_user_playlists(limit=50)['items']

    playlist_id = None
    for playlist in playlists:
        if playlist['name'].lower() == playlist_name.lower():
            playlist_id = playlist['id']
            break

    if not playlist_id:
        print("Playlist not found. Please check the name and try again.")
        return

    # Retrieve tracks from the playlist
    tracks = sp.playlist_tracks(playlist_id)['items']
    track_ids = [track['track']['id'] for track in tracks]
    track_names = [track['track']['name'] for track in tracks]

    # Retrieve Audio Features
    features_df = retrieve_audio_features(sp, track_ids)
    features_df['track_name'] = track_names

    # Handle missing values
    features_df = features_df.dropna()

    # Analyze mood based on valence
    avg_valence = features_df['valence'].mean()
    avg_energy = features_df['energy'].mean()

    if avg_valence > 0.7 and avg_energy > 0.6:
        print(f"The mood of the playlist '{playlist_name}' is mostly happy.")
    elif avg_valence > 0.5 and avg_energy <= 0.6:
        print(f"The mood of the playlist '{playlist_name}' is mostly calm.")
    elif avg_valence <= 0.5 and avg_energy < 0.4:
        print(f"The mood of the playlist '{playlist_name}' is mostly relaxed.")
    elif avg_valence <= 0.5 and avg_energy >= 0.4 and features_df['tempo'].mean() < 100:
        print(f"The mood of the playlist '{playlist_name}' is mostly melancholic.")
    else:
        print(f"The mood of the playlist '{playlist_name}' is mostly neutral.")

    # Provide context on why the mood was this way
    context = features_df[['danceability', 'energy', 'valence', 'tempo']].mean()
    print("\nMood context for the playlist:")
    print(f"Average Danceability: {context['danceability']:.2f}")
    print(f"Average Energy: {context['energy']:.2f}")
    print(f"Average Valence: {context['valence']:.2f}")
    print(f"Average Tempo: {context['tempo']:.2f}")


# Helper Function for Mood Classification
def classify_mood(row):
    valence = row['valence']
    energy = row['energy']
    tempo = row['tempo']

    if valence > 0.7 and energy > 0.6:
        return 'happy'
    elif valence > 0.5 and energy <= 0.6:
        return 'calm'
    elif valence <= 0.5 and energy < 0.4:
        return 'relaxed'
    elif valence <= 0.5 and energy >= 0.4 and tempo < 100:
        return 'melancholic'
    else:
        return 'neutral'

# Main Execution
if __name__ == "__main__":
    sp = authenticate_spotify()
    track_ids, track_names, timestamps = fetch_listening_history(sp)
    features_df = retrieve_audio_features(sp, track_ids)
    data, daily_data = preprocess_data(features_df, track_names, timestamps)
    model = build_and_evaluate_model(daily_data)
    user_options(daily_data, data, sp)

# General Tips
# Store data locally for repeated analysis
daily_data.to_csv('daily_data.csv')
activity_summary = data.groupby([data.index.date, 'track_name']).size().reset_index(name='listening_count')
activity_summary.to_csv('activity_summary.csv')
# TODO:
# 1. Advanced Data Visualization (Objective #3):
#    - Upgrade current plots to interactive charts using Plotly or Dash.
#    - Add visual insights, such as bar charts for playlist moods or heatmaps for listening frequency.

# 2. Model Enhancement (Objective #1):
#    - Add more features to the model, such as time-based features (e.g., day of the week).
#    - Implement feature selection to determine which features impact mood prediction the most.
#    - Conduct hyperparameter tuning using additional techniques like RandomizedSearchCV.

# 3. User Experience Improvement (Objective #5):
#    - Add CLI improvements, such as more descriptive prompts and error handling.
#    - Implement a user-friendly GUI (using Tkinter or PyQt) to replace the CLI for easier navigation.

# 4. Scalability and Data Management (Objective #2):
#    - Integrate SQLite to store user listening history and mood data, allowing for trend analysis over months.
#    - Update the data collection to store daily snapshots in the database.

# 5. Automation (Objective #6):
#    - Automate data collection by setting up a scheduled job to periodically update user listening history.
#    - Deploy the project (perhaps on a cloud platform like Heroku) for easy access across devices.

# 6. Advanced Mood Analysis:
#    - Use clustering (e.g., K-means) to group songs and moods, allowing the user to see if their playlists have distinct mood clusters.
