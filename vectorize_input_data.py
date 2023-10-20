import pandas as pd
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

parquet_file = 'objaverse/thingiverse/thingiverse.parquet'
df = pd.read_parquet(parquet_file, engine='pyarrow')
df_first_10 = df.head(10)

df_first_10['vector'] = df_first_10['metadata'].apply(vectorize)
print(df_first_10)

# specify path for export
path = 'database.txt'
# export DataFrame to text file
df_first_10[['fileIdentifier', 'metadata']].to_csv(path, sep=' ', index=False, header=False)


connections.connect(host='127.0.0.1', port='19530')

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
            FieldSchema(name="fileIdentifier", dtype=DataType.VARCHAR, is_primary=True, max_length=500),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="license", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="fileType", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="sha256", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=100),            
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description='search thingiverse')
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='vector', index_params=index_params)
    return collection

collection = create_milvus_collection('search_thingiverse', 1536)

insert_pipe = (pipe.input('df')
                   .flat_map('df', 'data', lambda df: df.values.tolist())
                   .map('data', 'res', ops.ann_insert.milvus_client(host='127.0.0.1', 
                                                                    port='19530',
                                                                    collection_name='search_thingiverse'))
                   .output('res')
)

insert_pipe(df_first_10)

collection.load()
print(collection.num_entities)