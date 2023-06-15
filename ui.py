import base64
import streamlit as st
import pandas as pd
from document_search import DocumentSearcher
from document_embedder import DocumentEmbedder

# Initialize the DocumentEmbedder and DocumentSearcher
embedder = DocumentEmbedder()
searcher = DocumentSearcher()

def show_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.title('Document Search Engine')

# File Upload
uploaded_files = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    with st.spinner('Processing and embedding uploaded files...'):
        embedder.embed_documents(uploaded_files)
    for uploaded_file in uploaded_files:
        show_pdf(uploaded_file)

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