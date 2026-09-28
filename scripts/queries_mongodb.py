from pymongo import MongoClient

client = MongoClient("mongodb://admin:admin123@localhost:27017/")

db = client["movie_intelligence"]
collection = db["movies"]
films = collection.find().limit(5)

print("\n 5 premier film ")

for film in films:
    print(
        film["movie_id"],
        "-",
        film["title"],
        "- note :",
        film["vote_average"]
    )


print("\n film note >8")

films = collection.find(
    {"vote_average": {"$gt": 8}},
    {"_id": 0, "title": 1, "vote_average": 1}
).limit(10)

for film in films:
    print(film)

print("\n film populaire")

films = collection.find(
    {"popularity": {"$gt": 50}},
    {"_id": 0, "title": 1, "popularity": 1}
).limit(10)

for film in films:
    print(film)


print("\n film anglais")

films = collection.find(
    {"original_language": "en"},
    {"_id": 0, "title": 1, "original_language": 1}
).limit(10)

for film in films:
    print(film)

print("\n nombre de film par language")

resultats = collection.aggregate([
    {
        "$group": {
            "_id": "$original_language",
            "nombre_films": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "nombre_films": -1
        }
    },
    {
        "$limit": 10
    }
])

for resultat in resultats:
    print(resultat)


print("\n===== NOMBRE DE FILMS PAR GENRE =====")

resultats = collection.aggregate([
    {
        "$unwind": "$genres"
    },
    {
        "$group": {
            "_id": "$genres",
            "nombre_films": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "nombre_films": -1
        }
    },
    {
        "$project": {
            "_id": 0,
            "genre": "$_id",
            "nombre_films": 1
        }
    }
])

for resultat in resultats:
    print(resultat)


client.close()