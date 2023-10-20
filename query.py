import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import ops, pipe, DataCollection
import openai

openai.api_key = ""

def vectorize(metadata):
    vector = openai.Embedding.create(
        input=metadata,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    return vector

connections.connect(host='127.0.0.1', port='19530')

fields = [
        FieldSchema(name="fileIdentifier", dtype=DataType.VARCHAR, is_primary=True, max_length=500),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="license", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="fileType", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="sha256", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=100),            
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields=fields, description='search thingiverse')

collection = Collection(name="search_thingiverse", schema=schema)
collection.load()

def search(text):
    # Search parameters for the index
    search_params={
        "metric_type": "L2"
    }

    results=collection.search(
        data=[vectorize(text)],  # Embeded search value
        anns_field="vector",  # Search across embeddings
        param=search_params,
        limit=1,  # Limit to five results per search
        output_fields=['fileIdentifier', 'fileType', 'metadata']
    )
    return results[0]

result = search('roof right')
print(result)