import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

from milvus import connect_milvus, get_milvus_schema, create_or_load_collection

# --- Milvus Setup ---
connect_milvus()
schema = get_milvus_schema()
collection = create_or_load_collection('legal_iq', schema)

# --- OpenAI Embeddings ---
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ['OPENAI_API_KEY']
)

# --- LangChain LLM ---
llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model="gpt-3.5-turbo"
)


def search_milvus(query, top_k=5):
    query_embedding = embedding_model.embed_query(query)
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_text", "section_title"]
    )
    docs = []
    for hit in results[0]:
        docs.append(Document(
            page_content=hit.entity.get("chunk_text"),
            metadata={
                "section_title": hit.entity.get("section_title")
            }
        ))
    return docs


def generate_response(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return response.content, docs


# --- Streamlit UI ---
st.title("UK Legislation QA Search")

query = st.text_input("Enter your legal question:")

if query:
    with st.spinner("Searching and generating answer..."):
        docs = search_milvus(query)
        answer, source_docs = generate_response(query, docs)

    st.subheader("Answer")
    st.write(answer)
    st.subheader("Source Chunks")
    for doc in source_docs:
        st.markdown(f"**Section:** {doc.metadata['section_title']}")
        st.write(doc.page_content)
        st.markdown("---")
