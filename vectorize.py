import openai
import pandas as pd
 
openai.api_key = ""
 
def vectorize(metadata):
    vector = openai.Embedding.create(
        input=metadata,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    return vector

def create_parquet(parquet_file, start, end):
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    new_df = df.iloc[start:end+1]
    new_df[['fileIdentifier', 'metadata']].to_csv('database.txt')
    new_df['vector'] = new_df['metadata'].apply(vectorize)
    new_df.to_parquet(f'parquet/thingiverse-{start}-{end}.parquet', engine='pyarrow')

parquet_file = 'objaverse/thingiverse/thingiverse.parquet'
for (start, end) in [(0,100), (100,200)]:
    create_parquet(parquet_file, start, end)
    print(f'finished {start} - {end}')