import pandas as pd
from pymilvus import connections, Collection
from towhee import ops, pipe


insert_pipe = (pipe.input('df')
                .flat_map('df', 'data', lambda df: df.values.tolist())
                .map('data', 'res', ops.ann_insert.milvus_client(host='127.0.0.1', 
                                                                    port='19530',
                                                                    collection_name='search_thingiverse'))
                .output('res')
)

def connect_db(collection_name):
    connections.connect(host='127.0.0.1', port='19530')
    collection = Collection(name=collection_name)
    return collection

def insert(start, end):
    parquet_file = f'parquet/thingiverse-{start}-{end}.parquet'
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    insert_pipe(df)

collection = connect_db('search_thingiverse')
for (start, end) in [(0,100), (100,200)]:
    insert(start, end)
    print(f'inserted {start} - {end}')

collection.load()
print(collection.num_entities)