## Quickstart with Milvus
Vectors, the output data format of Neutral Network models, can effectively encode information and serve a pivotal
role in AI applications such as knowledge base, semantic search, retrieval augmented generation (RAG) and more.
Milvus is an open-source vector database that suits AI applications of every size from running a demo chatbot in 
Jupyter notebook to building web-scale search that serves billions of users.

## Install Milvus
In this guide we use Milvus lite, a python library included in **pymilvus** that can be embedded into the client application.
Milvus also supports deployment on __Docker__ and __Kubernetes__for production use cases.
Install __pymilvus__ which contains both the python client library and Milvus Lite.

```
pip install -U pymilvus
```

## Set Up Vector Database
To create a local Milvus vector database, simply instantiate a **__MilvusClient__** by specifying a file name to store all data, 
such as "milvus_demo.db".

```
from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")
```


## Create a Collection
In Milvus, we need a collection to store vectors and their associated metadata. You can think of it as a table in
traditional SQL databases. When creating a collection, you can define schema and index params to configure vector specs 
such as dimensionality, index types and distant metrics.
There are also complex concepts to optimize the index for vector search performance.
At minimum, you only need to set the collection name and the dimentsion of the vector field of the collection.

```
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")

client.create_collection(
    collection_name="demo_collection",
    dimension=768
)
```

In the above setup,
    - The primary key and vector field use their default names ("id" and "vector")
    - The metric type (vector distance definition) is set to its default value (__COSINE__)
    - The primary key field accepts integers and does not automatically increments (namely not using auto-id feature).
    Alternatively, you can formally define the schema of the collection by following this [instruction]("https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Collections/create_schema.md")

## Prepare Data
In this guide, we use vector to perform semantic search on text.
We need to generate vectors for text by downloading embedding models.
This can be easily done by using the utility functions from **pymilvus[model]** library

```
pip install "pymilvus[model]"

```
Generate vector embeddings with default model. Milvus expects data to be inserted organized as a list of dictionaries, where each dictionary
represents a data record, termed as an entity.

```
from pymilvus import model

# If connection to https://huggingface.co/ failed, uncomment the following path
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created
print("Dim:", embedding_fn.dim, vectors[0].shape) # Dim: 768

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields:", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))
```

```
res = client.insert(collection_name="demo_collection", data=data)
print(res)
```

```
{'insert_count': 3, 'ids': [0,1,2], 'cost':0}
```

## Semantic Search
### Vector Search
Milvus accepts one or multipe vector search requests at the same time. The value of the query_vectors variable is a list of vectors,
where each vector is an array of float numbers

```
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",
    data=query_vectors,
    limit=2,
    output_fields=["text", "subject"], # specifies fields to be returned
)

print(res)
```

The output is a list of results, each mapping to a vector search query. Each query contains a list of results, where each result contains the entity
primary key, the distance to the query vector, and the entity details with specified **output_fields**


### Vector Search with Metadata Filtering
You can also conduct vector search while considering the values of the metadata (called "scalar" fields in Milvus, as scalar referes to non-vector data)
Done with a filter expression specifying certain criteria. Let's see how to search and filter with the **subject** field in the following example.

```
# Insert more docs in another subject.
docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]

vectors = embedding_fn.encode_documents(docs)
data = [
    {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
    for i in range(len(vectors))
]

client.insert(collection_name="demo_collection", data=data)

# This will exclude any text in "history" subject despite close to the query vector
res = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",
    limit=2,
    output_fields=["text", "subject"]
)

print(res)
```

By default, the scalar fields are not indexed. If you need to perform metadata filtered search in large dataset, you can consider 
using fixed schema and also turn on the [index]("https://milvus.io/docs/scalar_index.md") to improve the search performance.
In addtion to vector search, you can also perform other types of searches

### Query
A query() is an operation that retrieves all entries matching a criteria, such as a filter expression or matching some ids.

```
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'"
    output_fields = ["text", "subject"]
)
```

Directly retrieve entities by primary key:

```
res = client.query(
    collection_name="demo_collection",
    ids=[0,2],
    output_fields=["vector", "text", "subject"],
)
```

### Delete Entities
If you'd like to purge data, you can delete entities specifying the primary key or delete all entities matching a particular
filter expression.

```
# Delete entities by primary key
res = client.delete(collection_name="demo_collection", ids=[0,2])

print(res)

# Delete entities by a filter expression
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'biology'",
)

print(res)
```

### Load Existing Data
Since all data of Milvus Lite is stored in a local file, you can load all data into memory even after the program
terminates, by creating a **MilvusClient** with the existing file.
For example, this will recover the collections from "milvus_demo.db" file and continue to write data into it.

```
from pymilvus import MilvusClient
client = MilvusClient("milvus_demo.db")
```

### Drop the collection
If you would like to delete all the data in a collection, you can drop the collection with

```
#Drop collection
client.drop_collection(collection_name="demo_collection")
```

### Learn More
Milvus Lite is great for getting started with a local python program. If you have large scale data or would like to use
Milvus in production, you can learn about deploying Milvus on [Docker]("https://milvus.io/docs/install_standalone-docker.md")
and [Kubernetes]("https://milvus.io/docs/install_cluster-milvusoperator.md").
All deployment modes of Milvus share the same API, so your client side code doesn't need to change much if moving to another 
deployment mode. Simply specify the **__URI and Token__** of a Milvus serves deployed anywhere.

```
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
```