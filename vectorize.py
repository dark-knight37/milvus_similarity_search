import openai
 
openai.api_key = ""
 
article = '{"filename": "Mighty_Habbi-Kieran2.stl"}'
 
vector = openai.Embedding.create(
    input=article,
    model="text-embedding-ada-002"
)["data"][0]["embedding"]

print(vector)