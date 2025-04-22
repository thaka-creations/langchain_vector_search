import psycopg2
import numpy as np

# connect to the database
conn = psycopg2.connect(
    host="localhost", database="masomo", user="postgres", password=""
)
cur = conn.cursor()

# Insert a vector
embedding = np.array([1.5, 2.5, 3.5])
cur.execute("INSERT INTO items (embedding) VALUES (%s)", (embedding.tolist(),))

# Perform a similarity search
query_vector = np.array([2, 3, 4])
cur.execute(
    "SELECT * FROM items ORDER BY embedding <-> %s::vector LIMIT 1",
    (query_vector.tolist(),),
)
result = cur.fetchone()
print(f"Nearest neighbor: {result}")

conn.commit()
cur.close()
conn.close()
