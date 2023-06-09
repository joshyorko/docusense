import base64
import streamlit as st
import pandas as pd
from documentsearch import DocumentSearcher
from documentembedder import DocumentEmbedder

# Initialize the DocumentEmbedder and DocumentSearcher
embedder = DocumentEmbedder()
searcher = DocumentSearcher()

st.title('Document Search Engine')

# File Upload
uploaded_files = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    with st.spinner('Processing and embedding uploaded files...'):
        embedder.embed_documents(uploaded_files)

# Search Query
query = st.text_input("Enter your search query")
if query:
    with st.spinner('Searching documents...'):
        results = searcher.search(query)
        if results.empty:
            st.write('No results found')
        else:
            st.dataframe(results)

# Optional: Download Results as CSV
if not results.empty and st.button('Download Results as CSV'):
    csv = results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="search_results.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)