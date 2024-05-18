import os
import openai
# import torch
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, PromptTemplate, get_response_synthesizer, Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.extractors import (TitleExtractor, QuestionsAnsweredExtractor)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display
from transformers import AutoTokenizer


os.environ['OPENAI_API_KEY'] = 'sk-...'
openai.api_key = os.environ['OPENAI_API_KEY']
input_dir_path = '/Users/marmik/test/'
cohere_api_key = '...'

# loading the data
documents = SimpleDirectoryReader(input_dir = input_dir_path,
                                  required_exts=['.pdf'],
                                  recursive = True).load_data(show_progress=True)

# custom transformations
Settings.chunk_size = 512 
chunk_overlap = 20 # no. of chars overlap b/w two consecutive chunks

text_splitter = SentenceSplitter(chunk_size = Settings.chunk_size, chunk_overlap = chunk_overlap)
# text_splitter = TokenTextSplitter(separator = ' ', chunk_size = chunk_size, chunk_overlap=chunk_overlap)

title_extractor = TitleExtractor(nodes=10)
qa_extractor = QuestionsAnsweredExtractor(questions=5)

# embedding model
embed_model = OpenAIEmbedding(model = 'text-embedding-3-large', embed_batch_size = 42,
                              api_key = os.environ['OPENAI_API_KEY'])

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
# vector_index = VectorStoreIndex.from_vector_store(vector_store, 
#                                                   embed_model=embed_model,
#                                                   storage_context = storage_context)


# tokenizer model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# conditions when the text generation should stop
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
# wrap the default prompts that are internal to llam_index
query_wrapper_prompt = PromptTemplate('<|USER|>{query_str}<|ASSISTANT|')

system_prompt = 'You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.'


# setting up the llm

llm = OpenAI(model="gpt-4o", temperature=0.5, max_tokens=256)

# llm = HuggingFaceLLM(
#     context_window = 8192,
#     max_new_tokens = 256,
#     generate_kwargs = {'temperature':0.7, 'do_sample':False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
#     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#     device_map="auto",
#     stopping_ids=stopping_ids,
#     tokenizer_kwargs={'max_length' : 4096},
#     # model_kwargs={'torch_dtype' : torch.float16},

# )

Settings.llm = llm

# configure retriever
# retriever = VectorIndexRetriever(
#     index= vector_index,
#     similarity_top_k = 10
# )

# configure response synthesizer
response_synthesizer = get_response_synthesizer(llm=Settings.llm,)

# reranker model
reranker = CohereRerank(api_key = cohere_api_key, top_n = 3)

# assemble query engine
# query_engine = RetrieverQueryEngine.from_args(
#     llm = llm,
#     retriever = retriever,
#     response_synthesizer=response_synthesizer,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff = 0.7), reranker],
#     streaming = True,
#     )

query_engine = vector_index.as_query_engine(streaming=True)

# a custom prompt template to refine responses from LLM
qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts({"response_synthesizer:text_qa_template" : qa_prompt_tmpl})

query_str = 'What is the given paper about?'

response = query_engine.query(query_str)
print(str(response))
