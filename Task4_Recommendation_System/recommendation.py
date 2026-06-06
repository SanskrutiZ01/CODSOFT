import csv

movies = []

# Load movie data
with open("Task4_Recommendation_System/movies.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        movies.append(row)

# Title
print("🎬 Welcome to Movie & Anime Recommendation System")
print("Get recommendations based on your interests.\n")

# User input
industry = input(
    "Enter Industry (Hollywood / Bollywood / Korean / Japanese / Chinese / Thai): "
).strip()

genre = input(
    "Enter Genre (Action / Romance / Drama / Thriller / Comedy / Sci-Fi / Anime): "
).strip()

# Find matches
recommendations = []

for movie in movies:
    if (
        movie["industry"].lower() == industry.lower()
        and movie["genre"].lower() == genre.lower()
    ):
        recommendations.append(movie["title"])

# Output
print("\n🎥 Recommended for you:\n")

if recommendations:
    for movie in recommendations:
        print("-", movie)
else:
    print("No matching recommendations found.")