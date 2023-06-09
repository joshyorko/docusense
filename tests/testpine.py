import os
import pinecone

# get api key from your environment or replace 'PINECONE_API_KEY' with your actual key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
# get your environment from your environment or replace 'PINECONE_ENVIRONMENT' with your actual environment
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

pinecone.init(api_key='f6ff4ba3-0003-4171-a3f5-8504143e990f', environment='northamerica-northeast1-gcp')

# list indexes
indexes = pinecone.list_indexes()

print("Indexes: ", indexes)
