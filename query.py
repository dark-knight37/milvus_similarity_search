from pymilvus import connections, Collection
import openai
import objaverse.xl as oxl
import pandas as pd

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

def search(collection, text, num):
    # Search parameters for the index
    search_params={
        "metric_type": "L2"
    }

    results=collection.search(
        data=[vectorize(text)],  # Embeded search value
        anns_field="vector",  # Search across embeddings
        param=search_params,
        limit=num,  # Limit to five results per search
        output_fields=['fileIdentifier', 'fileType', 'metadata']
    )
    
    result = []
    for hit in results[0]:
        result.append(hit.id)

    print("finished query")
    return result

def download_obj(parquet_file, identifier, dest):
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    result = df[df['fileIdentifier'].isin(identifier)]
    print('finished dataframe')
    # oxl.download_objects(
    #     objects = result,
    #     download_dir = dest
    # )
    # print('finish downloading')
    return result

openai.api_key = ""
parquet_file = 'objaverse/thingiverse/thingiverse.parquet'
dest = 'objects'
object_description = 'roof'
download_num = 1
collection = connect_db('search_thingiverse')
identifier = search(collection, object_description, download_num)
result = download_obj(parquet_file, identifier, dest)
print(result)