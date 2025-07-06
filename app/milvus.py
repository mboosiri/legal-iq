import os

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


def connect_milvus():
    """Establish connection to Milvus."""
    try:
        connections.connect(
            alias='default',
            uri=os.environ['MILVUS_CLUSTER_ENDPOINT'],
            token=os.environ['MILVUS_API_KEY']
        )
        # Check if the server is ready
        server_version = utility.get_server_version()
        print(f"Connected to Milvus server version: {server_version}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit(1)


def create_or_load_collection(collection_name: str, schema: CollectionSchema):
    """
    Create a Milvus collection if it does not exist; otherwise, load the existing collection.
    """
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Loading collection.")
        collection = Collection(name=collection_name)
    else:
        print(f"Creating collection '{collection_name}'.")
        collection = Collection(name=collection_name, schema=schema)
    return collection


# Define Milvus collection
def get_milvus_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="section_title", dtype=DataType.VARCHAR, dim=512),
        FieldSchema(name="doc_url", dtype=DataType.VARCHAR, dim=255),
    ]
    schema = CollectionSchema(fields, description="UK Legislation Embeddings")
    return schema
