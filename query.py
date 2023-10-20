from pymilvus import connections, Collection
import openai

openai.api_key = ""

def vectorize(metadata):
    vector = openai.Embedding.create(
        input=metadata,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    return vector

def connect_db(collection_name):
    connections.connect(host='127.0.0.1', port='19530')
    collection = Collection(name=collection_name)
    return collection

def search(collection, text):
    # Search parameters for the index
    search_params={
        "metric_type": "L2"
    }

    results=collection.search(
        data=[vectorize(text)],  # Embeded search value
        anns_field="vector",  # Search across embeddings
        param=search_params,
        limit=3,  # Limit to five results per search
        output_fields=['fileIdentifier', 'fileType', 'metadata']
    )
    
    result = []
    for hit in results[0]:
        result.append(hit.id)

    return result

collection = connect_db('search_thingiverse')
result = search(collection, 'tank')
print(result)