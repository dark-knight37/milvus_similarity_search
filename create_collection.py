from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

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

# Create collection
collection = create_milvus_collection('search_thingiverse', 1536)

collection.load()
print(collection.num_entities)