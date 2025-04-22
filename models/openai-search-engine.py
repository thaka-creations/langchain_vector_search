import numpy as np
import psycopg2
from openai import OpenAI

client = OpenAI()

# Connect to the database
conn = psycopg2.connect(
    host="localhost", database="masomo", user="postgres", password=""
)
cur = conn.cursor()

# Create a table for our documents
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(1536)
    )
    """
)


# Function to get embeddings from OpenAI
def get_embeddings(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Function  to add a document
def add_document(content):
    embedding = get_embeddings(content)
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, embedding),
    )
    conn.commit()


# Function to search for documents
def search_documents(query, limit=5):
    query_embedding = get_embeddings(query)
    cur.execute(
        """
        SELECT content, embedding <-> %s::vector AS distance
        FROM documents
        ORDER BY distance
        LIMIT %s
    """,
        (query_embedding, limit),
    )
    return cur.fetchall()


# Add some sample documents
sample_docs = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I've got a lovely bunch of coconuts",
    "The brown fox is quick and the blue dog is lazy!",
    "The quick brown fox jumps over the lazy dog",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I've got a lovely bunch of coconuts",
    "The brown fox is quick and the blue dog is lazy!",
    "The quick brown fox jumps over the lazy dog.",
    "Python is a high-level programming language.",
    "Vector databases are essential for modern AI applications.",
    "PostgreSQL is a powerful open-source relational database.",
]

for doc in sample_docs:
    add_document(doc)

# Perform a search
query = "Something about python"
results = search_documents(query)

print(f"Search results for '{query}':")
for i, (content, distance) in enumerate(results, 1):
    print(f"{i}. {content} (Distance: {distance:.4f})")

# Close the connection
cur.close()
conn.close()
