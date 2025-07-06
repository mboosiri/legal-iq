import os

from tqdm import tqdm

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from milvus import connect_milvus, get_milvus_schema, create_or_load_collection

# Initialize OpenAI
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ['OPENAI_API_KEY']
)


def get_legislation_documents(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


# --- Split Text using LangChain ---
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# --- Main process ---
def process_legislation_page(url):
    documents = get_legislation_documents(url)
    chunks = chunk_documents(documents)

    to_insert = []
    for chunk in tqdm(chunks, desc="Embedding and inserting"):
        text = chunk.page_content
        metadata = chunk.metadata
        embedding = embedding_model.embed_query(text)

        # For demonstration: pull a simple section title if available
        section_title = metadata.get("title", "Unknown Section")

        to_insert.append([
            text,
            embedding,
            section_title,
            url
        ])

    # Insert to Milvus
    if to_insert:
        chunk_texts, embeddings, section_titles, urls = zip(*to_insert)
        collection.insert([
            list(embeddings),
            list(urls),
            list(section_titles),
            list(chunk_texts),
        ])


# --- Connect Milvus ---
connect_milvus()

# --- Create/Load Collection ---
schema = get_milvus_schema()
collection = create_or_load_collection('legal_iq', schema)

# --- Example URL ---
legislation_url = "https://www.legislation.gov.uk/ukpga/2018/12/contents/enacted"
process_legislation_page(legislation_url)
