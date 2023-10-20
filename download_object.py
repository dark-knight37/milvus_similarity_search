import pandas as pd
import objaverse.xl as oxl

df = pd.read_parquet('objaverse/thingiverse/thingiverse.parquet', engine='pyarrow')
# df = pd.read_parquet('objaverse/sketchfab/sketchfab.parquet', engine='pyarrow')
# df = pd.read_parquet('objaverse/smithsonian/smithsonian.parquet', engine='pyarrow')
# df = pd.read_parquet('objaverse/github/github.parquet', engine='pyarrow')
sampled_df = df[1001:1010]
oxl.download_objects(sampled_df, download_dir='objects')

# save metadata into a file
with open('objaverse.txt', 'w', encoding='UTF-8') as f:
    df_string = sampled_df.to_string(header=True, index=False)
    f.write(df_string)
    