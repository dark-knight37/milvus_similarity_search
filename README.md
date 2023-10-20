# milvus_similarity_search
similarity search using milvus database

## How to install milvus on ubuntu 22.04 and set environment
Deb package
~~~bash
wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus_2.1.4-1_amd64.deb
~~~

Install package
~~~bash
sudo apt-get update
sudo dpkg -i milvus_2.1.4-1_amd64.deb
sudo apt-get -f install
~~~

Install python packages
~~~bash
pip install -q towhee pymilvus==2.1.0
pip install milvus_cli==0.3.0
~~~

## sample
sample code is in sample folder.
- reference: https://github.com/towhee-io/examples/blob/main/nlp/text_search/search_article_in_medium.ipynb

## objaverse
- Download thingiverse.parquet file from objaverse
- vectorize.py : create pieces of parquet and vectorize each pieces
- create_collection.py : create collection in milvus database
- input_data.py : input parquet data into database
- query.py : query from database