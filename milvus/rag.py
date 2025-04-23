"""
Retrieval Augmented Generation (RAG) Pipeline with Milvus

A RAG system enhances large language model outputs by incorporating relevant knowledge from a document collection:

1. Document Retrieval: When given a prompt, the system first searches a vector database (Milvus)
   to find the most semantically similar documents from the corpus (a collection of text documents that have been processed and stored in the vector database)

2. Context Augmentation: The retrieved documents are used as additional context to inform
   the language model's response

3. Text Generation: The language model generates a response that combines its base knowledge
   with the specific information found in the retrieved documents

This approach improves accuracy and grounds the model's outputs in source documentation.
"""

# Import glob module to find pathnames matching a pattern
# Used for finding files that match specified patterns
import json
import textwrap
from glob import glob

from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm

client = OpenAI()

text_lines = []

for file_path in glob("milvus_docs/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")
    print(text_lines)


# Function to get embeddings from OpenAI
def get_embeddings(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Generate a test embedding and print its dimension and first few elements
test_embedding = get_embeddings("This is a test")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")
print(f"First few elements: {test_embedding[:10]}")


# Load data into Milvus
# create collection
milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

# check if the collection exists and drop it if it does
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

# create collection
# if we don't specify any field information, milvus will automatically create
# a default id field for primary key, and a vector field to store the vector data.
# A reserved JSON field is used to store non-schema-defined fields and their values.

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

# insert data
data = []
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append(
        {
            "id": i,
            "vector": get_embeddings(line),
            "text": line,
        }
    )

# insert data into collection
milvus_client.insert(collection_name=collection_name, data=data)

# query data
## Build RAG
### Retrieve data from a query
question = "How is data stored in milvus?"

# search for the question in the collection and retrieve semantically similar top 3 results
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[get_embeddings(question)],
    limit=3,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["text"],
)
print(search_res)

retrieved_lines_with_distance = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]

print(json.dumps(retrieved_lines_with_distance, indent=4))

## Use LLM to get a RAG response
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distance]
)
print("question: \n", question)
print("context: \n", context)

# Define system and user prompts for Large Model.
# This prompt is assembled with the retrieved documents from Milvus.
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

# Use OpenAI to generate a response
response = client.responses.create(
    model="gpt-4.1-nano", instructions=SYSTEM_PROMPT, input=USER_PROMPT
)
print(textwrap.fill(response.output_text, width=80))
