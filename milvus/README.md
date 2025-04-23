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


