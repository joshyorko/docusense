import os
import pandas as pd
from pdf2image.pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pytesseract
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pinecone



# Initialize Pinecone
pinecone.init(api_key='f6ff4ba3-0003-4171-a3f5-8504143e990f', environment='northamerica-northeast1-gcp')

# Define your Pinecone index name
pinecone_index_name = "document-search"

# Create Pinecone index
if pinecone_index_name not in pinecone.list_indexes():
    pinecone.create_index(
    name=pinecone_index_name, 
    dimension=384, 
    metric="cosine", 
    shards=1
)

index = pinecone.Index(index_name=pinecone_index_name)

def get_info(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        info = pdf.metadata
    except Exception as e:
        print(f"Failed to extract metadata from {pdf_path}: {e}")
        info = {}
    return info

def preprocess_image(image):
    return image.convert('L')

def pdf_to_text(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        text = ""
        for i in range(len(images)):
            image = preprocess_image(images[i])
            text += pytesseract.image_to_string(image, lang='eng')
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
        text = ""
    return text

def process_pdf(pdf_path):
    info = get_info(pdf_path)
    text = pdf_to_text(pdf_path)
    return info, text.strip()

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # THIS IS WHERE WE WILL BUILD IN THE FUNCTIONALITY FOR FILE UPLOAD
    dir_path = "sample_pdfs/test2"
    pdf_files = [f for f in os.listdir(dir_path) if f.endswith(".pdf")]
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    batch_size = 100  # adjust as needed
    num_batches = len(pdf_files) // batch_size + (len(pdf_files) % batch_size != 0)

    for b in tqdm(range(num_batches)):
        batch_files = pdf_files[b*batch_size:(b+1)*batch_size]
        ids = []
        embeddings = []
        metadatas = []
        df_data = []
        for pdf_file in batch_files:
            pdf_path = os.path.join(dir_path, pdf_file)
            info, text = process_pdf(pdf_path)
            df_dict = {'File': pdf_file, 'Metadata': info, 'Extracted_Text': text}
            df = pd.json_normalize(df_dict)
            df.columns = df.columns.str.replace('Metadata.', '').str.replace('/','')
            df['Extracted_Text'] = df['Extracted_Text'].str.replace('\n', ' ')
            #df.to_csv('extracted_text.csv', mode='a', header=False')
            df_data.append(df)
        
            # Calculate embeddings for the entire row and store them for batch upsert
            row_text = " ".join(str(val) for val in df.iloc[0])
            embedding = model.encode([row_text])
            ids.append(pdf_file)
            embeddings.append(embedding.tolist())
            # Store the entire row in the metadata
            metadatas.append(df.iloc[0].to_dict())
        # Upsert the batch of embeddings
        index.upsert(vectors=zip(ids, embeddings, metadatas))
        df_ = pd.concat(df_data,ignore_index=True)['Extracted_Text'].to_list()
        with open('extracted_text.txt', 'w') as f:
            for item in df_:
                f.write("%s\n" % item)
        return 
    
if __name__ == "__main__":
    main()


