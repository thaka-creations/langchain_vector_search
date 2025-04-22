"""
This example assumes you have set up OpenAI embeddings and have a list of documents to embed.
"""

from langchain_postgres.vectorstores import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings


# Set up the connection string and embedding function
connection_string = "postgresql://user:pass@localhost:5432/db_name"
embedding_function = OpenAIEmbeddings()

# Create a PGVector instance
vector_store = PGVector.from_documents(
    documents, embedding_function, connection_string=connection_string
)

# Perform a similarity search
query = "Your query here"
results = vector_store.similarity_search(query)
