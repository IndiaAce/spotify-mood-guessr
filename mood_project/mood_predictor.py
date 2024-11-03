from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
        scope="user-read-recently-played user-library-read"
    ))

# Data Collection and Preprocessing
def collect_and_preprocess_data(sp):
    # Fetch Listening History
    results = sp.current_user_recently_played(limit=50)
    track_ids = [item['track']['id'] for item in results['items']]
    track_names = [item['track']['name'] for item in results['items']]

    # Retrieve Audio Features
    features = sp.audio_features(tracks=track_ids)

    # Organize data into a DataFrame
    data = pd.DataFrame(features)
    data['timestamp'] = [item['played_at'] for item in results['items']]
    data['track_name'] = track_names

    # Handle missing values
    data = data.dropna()

    # Time Aggregation
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Only aggregate numeric columns
    daily_data = data.select_dtypes(include=[np.number]).resample('D').mean()

    # Mood Labeling
    daily_data['mood_label'] = np.where(daily_data['valence'] > 0.5, 'positive', 'negative')

    # Feature Engineering
    daily_data['energy_median'] = data['energy'].resample('D').median()
    daily_data['tempo_sum'] = data['tempo'].resample('D').sum()

    return data, daily_data

# Model Building and Evaluation
def build_and_evaluate_model(daily_data):
    X = daily_data[['energy', 'tempo', 'danceability', 'valence', 'energy_median', 'tempo_sum']]  # Features
    y = daily_data['mood_label']  # Labels

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training with refined hyperparameters
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

# User Interaction Options
def user_options(daily_data, data):
    while True:
        print("\nSelect an option:")
        print("1. See overall mood trend")
        print("2. Check your current mood")
        print("3. Look at your mood based on songs you listened to last week")
        print("4. See your top played songs in the last week and their mood")
        print("5. Exit")
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
    top_tracks_data['mood_label'] = np.where(top_tracks_data['valence'] > 0.5, 'positive', 'negative')

    print("\nTop Played Songs in Last Week and Their Mood:")
    for track_name, count in top_tracks.items():
        mood = top_tracks_data[top_tracks_data['track_name'] == track_name]['mood_label'].iloc[0]
        print(f"Track Name: {track_name} - Mood: {mood} - Play Count: {count}")

    # Save top played songs to CSV
    top_tracks_data[['track_name', 'mood_label']].to_csv('top_played_songs_last_week.csv', index=False)

# Main Execution
if __name__ == "__main__":
    sp = authenticate_spotify()
    data, daily_data = collect_and_preprocess_data(sp)
    model = build_and_evaluate_model(daily_data)
    user_options(daily_data, data)

# General Tips
# Store data locally for repeated analysis
daily_data.to_csv('daily_data.csv')
activity_summary = data.groupby([data.index.date, 'track_name']).size().reset_index(name='listening_count')
activity_summary.to_csv('activity_summary.csv')

# Automate data collection for regular intervals
# TODO: Set up a scheduler (e.g., cron job) to run data collection periodically

# Ensure data privacy and ethical considerations
# TODO: Encrypt stored data and ensure access control policies are in place

# Instructions for Secure Setup
# - Create a .env file in the root of your project with the following variables:
#   SPOTIFY_CLIENT_ID=your_client_id
#   SPOTIFY_CLIENT_SECRET=your_client_secret
#   SPOTIFY_REDIRECT_URI=your_redirect_uri
# - Add .env to your .gitignore file to avoid exposing sensitive credentials

# Next Steps
# - Define specific objectives for the project
# - Set milestones and gather initial data for testing
# - Build initial models and iterate based on findings
