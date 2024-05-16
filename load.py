import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.extractors import (TitleExtractor, QuestionsAnsweredExtractor)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ['OPENAI_API_KEY'] = 'sk-...'

# loading the data
documents = SimpleDirectoryReader('./data').load_data(show_progress=True)

# custom transformations
chunk_size = 512 # max size of each chunk
chunk_overlap = 10 # no. of chars overlap b/w two consecutive chunks

text_splitter = SentenceSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
# text_splitter = TokenTextSplitter(separator = ' ', chunk_size = chunk_size, chunk_overlap=chunk_overlap)

title_extractor = TitleExtractor(nodes=10)
qa_extractor = QuestionsAnsweredExtractor(questions=5)

embed_model = OpenAIEmbedding(model = 'text-embedding-3-large', embed_batch_size = 42)

# global
Settings.text_splitter = text_splitter
Settings.embed_model = embed_model

# storing the embeddings in a Vector Store

# initalize the chroma client
db = chromadb.PersistentClient(path = './chroma_db')

# create collection 
chroma_collection = db.get_or_create_collection('quickstart')

# assign chroma as vector store in a StorageContext
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store = vector_store)

# create index
vector_index = VectorStoreIndex.from_documents(documents, 
                                               transformations = [text_splitter, title_extractor, qa_extractor],
                                               storage_context = storage_context,
                                               embed_model = embed_model,
                                               show_progress = True)

# load index from stored vectors
vector_index = VectorStoreIndex.from_vector_store(vector_store, 
                                                  embed_model=embed_model,
                                                  storage_context = storage_context)

# embed the query text
query_engine = vector_index.as_query_engine()

embeddings = embed_model.get_text_embedding('YOOO THIS IS WORKING !? ')

print(embeddings[:5])
