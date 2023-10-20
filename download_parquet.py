import objaverse.xl as oxl

annotations = oxl.get_annotations(download_dir="/objaverse")

print(annotations["source"].value_counts())
print(annotations["fileType"].value_counts())

# sample a single object from each source
# sampled_df = annotations.groupby('source').apply(lambda x: x.sample(10)).reset_index(drop=True)
# oxl.download_objects(objects=sampled_df)