import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import ops, pipe, DataCollection

connections.connect(host='127.0.0.1', port='19530')

search_pipe = (pipe.input('query')
                    .map('query', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
                    .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
                    .flat_map('vec', ('id', 'score', 'title', 'link'), ops.ann_search.milvus_client(host='127.0.0.1', 
                                                                                   port='19530',
                                                                                   collection_name='search_article_in_medium',
                                                                                   output_fields=['title', 'link']))  
                    .output('query', 'id', 'score', 'title', 'link')
               )

res = search_pipe('funny story')
DataCollection(res).show()