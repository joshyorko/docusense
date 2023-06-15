import pandas as pd
from pdf2image.pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
import pytesseract
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pinecone
import io


class DocumentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', api_key='', 
                 environment='northamerica-northeast1-gcp', index_name="document-search", 
                 tesseract_cmd='/usr/bin/tesseract'):
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        self.model = SentenceTransformer(model_name)
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name, 
                dimension=384, 
                metric="cosine", 
                shards=1
            )
        self.index = pinecone.Index(index_name=self.index_name)

        # Initialize the S3 client
       

    def get_info(self, pdf_file):
        try:
            pdf = PdfFileReader(pdf_file)
            info = pdf.getDocumentInfo()
        except Exception as e:
            print(f"Failed to extract metadata: {e}")
            info = {}
        return info

    def preprocess_image(self, image):
        return image.convert('L')

    def pdf_to_text(self, pdf_file):
        try:
            images = convert_from_path(pdf_file, dpi=300)
            text = ""
            for i in range(len(images)):
                image = self.preprocess_image(images[i])
                text += pytesseract.image_to_string(image, lang='eng')
        except Exception as e:
            print(f"Failed to extract text: {e}")
            text = ""
        return text



    def process_pdf(self, pdf_file):
        info = self.get_info(pdf_file)
        text = self.pdf_to_text(pdf_file)
        return info, text.strip()

    def embed_documents(self, uploaded_files, batch_size=100):
        num_batches = len(uploaded_files) // batch_size + (len(uploaded_files) % batch_size != 0)

        for b in tqdm(range(num_batches)):
            batch_files = uploaded_files[b*batch_size:(b+1)*batch_size]
            ids = []
            embeddings = []
            metadatas = []
            df_data = []
            for pdf_file in batch_files:
                pdf_path = io.BytesIO(pdf_file.getbuffer())
                info, text = self.process_pdf(pdf_path)
                df_dict = {'File': pdf_file.name, 'Metadata': info, 'Extracted_Text': text}
                df = pd.json_normalize(df_dict)
                df.columns = df.columns.str.replace('Metadata.', '').str.replace('/','')
                df['Extracted_Text'] = df['Extracted_Text'].str.replace('\n', ' ')
                df_data.append(df)

                row_text = " ".join(str(val) for val in df.iloc[0])
                embedding = self.model.encode([row_text])
                ids.append(pdf_file.name)
                embeddings.append(embedding.tolist())
                metadatas.append(df.iloc[0].to_dict())

            self.index.upsert(vectors=zip(ids, embeddings, metadatas))
            df_ = pd.concat(df_data,ignore_index=True)['Extracted_Text']
