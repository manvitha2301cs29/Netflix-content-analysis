import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load the dataset
print("Loading Netflix dataset")
df = pd.read_csv("netflix_titles.csv")
print(f"Original dataset shape: {df.shape}")

# Show missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Clean data - more conservative approach
df.dropna(subset=['type', 'title'], inplace=True)
df['country'] = df['country'].fillna('Unknown')
df['listed_in'] = df['listed_in'].fillna('Unknown')
df['date_added'] = df['date_added'].fillna('Unknown')

# Clean and convert dates
df['date_added_clean'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df_with_dates = df[df['date_added_clean'].notna()].copy()
df_with_dates['year_added'] = df_with_dates['date_added_clean'].dt.year

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Rows with valid dates: {len(df_with_dates)}")

# 1. Titles added per year
titles_per_year = df_with_dates['year_added'].value_counts().sort_index()

# 2. Content type distribution
type_counts = df['type'].value_counts()

# 3. Top countries (simple approach - first country only)
df['first_country'] = df['country'].apply(lambda x: x.split(',')[0].strip() if x != 'Unknown' else 'Unknown')
top_countries = df['first_country'].value_counts().head(10)

# 4. Top genres (simple approach - first genre only)
df['first_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip() if x != 'Unknown' else 'Unknown')
top_genres = df['first_genre'].value_counts().head(10)

# 5. Content ratings
rating_counts = df['rating'].value_counts().head(8)

# Create visualization with better spacing
fig, axs = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('Netflix Content Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)

# Plot 1: Titles Added Per Year
if len(titles_per_year) > 0:
    titles_per_year.plot(kind='bar', color='tomato', ax=axs[0, 0])
    axs[0, 0].set_title("Titles Added Per Year", fontsize=12, pad=20)
    axs[0, 0].set_xlabel("Year", fontsize=10)
    axs[0, 0].set_ylabel("Number of Titles", fontsize=10)
    axs[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
    axs[0, 0].tick_params(axis='y', labelsize=9)

# Plot 2: Movies vs TV Shows (Pie Chart)
colors = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = axs[0, 1].pie(type_counts.values, labels=type_counts.index, 
                                        autopct='%1.1f%%', colors=colors, startangle=90,
                                        textprops={'fontsize': 10})
axs[0, 1].set_title("Movies vs TV Shows", fontsize=12, pad=20)

# Plot 3: Top 10 Countries
top_countries_clean = top_countries[top_countries.index != 'Unknown'][:10]
if len(top_countries_clean) > 0:
    top_countries_clean.plot(kind='barh', color='skyblue', ax=axs[0, 2])
    axs[0, 2].set_title("Top 10 Countries", fontsize=12, pad=20)
    axs[0, 2].set_xlabel("Number of Titles", fontsize=10)
    axs[0, 2].tick_params(axis='x', labelsize=9)
    axs[0, 2].tick_params(axis='y', labelsize=9)
    # Ensure country names don't overlap
    axs[0, 2].yaxis.set_tick_params(pad=5)

# Plot 4: Top 10 Genres
top_genres_clean = top_genres[top_genres.index != 'Unknown'][:10]
if len(top_genres_clean) > 0:
    bars = axs[1, 0].bar(range(len(top_genres_clean)), top_genres_clean.values, 
                        color='purple', alpha=0.7)
    axs[1, 0].set_title("Top 10 Genres", fontsize=12, pad=20)
    axs[1, 0].set_xticks(range(len(top_genres_clean)))
    
    # Truncate long genre names to prevent overlap
    genre_labels = [label[:15] + '...' if len(label) > 15 else label for label in top_genres_clean.index]
    axs[1, 0].set_xticklabels(genre_labels, rotation=45, ha='right', fontsize=9)
    axs[1, 0].set_ylabel("Count", fontsize=10)
    axs[1, 0].tick_params(axis='y', labelsize=9)

# Plot 5: Content by Rating
if len(rating_counts) > 0:
    rating_counts.plot(kind='bar', color='orange', ax=axs[1, 1])
    axs[1, 1].set_title("Content by Rating", fontsize=12, pad=20)
    axs[1, 1].set_xlabel("Rating", fontsize=10)
    axs[1, 1].set_ylabel("Count", fontsize=10)
    axs[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
    axs[1, 1].tick_params(axis='y', labelsize=9)

# Plot 6: Movie Duration Distribution
if 'duration' in df.columns:
    movies_df = df[df['type'] == 'Movie'].copy()
    # Extract minutes from duration (e.g., "90 min" -> 90)
    movies_df['duration_minutes'] = movies_df['duration'].str.extract(r'(\d+)').astype(float)
    
    if not movies_df['duration_minutes'].isna().all():
        axs[1, 2].hist(movies_df['duration_minutes'].dropna(), bins=20, 
                      color='lightcoral', alpha=0.7, edgecolor='black')
        axs[1, 2].set_title("Movie Duration Distribution", fontsize=12, pad=20)
        axs[1, 2].set_xlabel("Duration (minutes)", fontsize=10)
        axs[1, 2].set_ylabel("Frequency", fontsize=10)
        axs[1, 2].tick_params(axis='x', labelsize=9)
        axs[1, 2].tick_params(axis='y', labelsize=9)

# Improve layout with better spacing
plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)

# Print summary statistics
print("\n" + "="*50)
print("NETFLIX CONTENT ANALYSIS SUMMARY")
print("="*50)
print(f" Total titles analyzed: {len(df):,}")
print(f" Movies: {type_counts.get('Movie', 0):,} ({type_counts.get('Movie', 0)/len(df)*100:.1f}%)")
print(f" TV Shows: {type_counts.get('TV Show', 0):,} ({type_counts.get('TV Show', 0)/len(df)*100:.1f}%)")

if len(top_countries_clean) > 0:
    print(f" Top country: {top_countries_clean.index[0]} ({top_countries_clean.iloc[0]} titles)")

if len(top_genres_clean) > 0:
    print(f" Top genre: {top_genres_clean.index[0]} ({top_genres_clean.iloc[0]} titles)")

if len(titles_per_year) > 0:
    peak_year = titles_per_year.idxmax()
    print(f" Peak year for additions: {peak_year} ({titles_per_year.max()} titles)")

print("="*50)

# Save and show
plt.savefig('netflix_analysis_simple.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Analysis completed successfully!")
print(" Chart saved as 'netflix_analysis_simple.png'")