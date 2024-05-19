import streamlit as st
import uuid
import tempfile
import base64
import gc
import os
import openai
# import torch
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, PromptTemplate, get_response_synthesizer, Settings
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.extractors import (TitleExtractor, QuestionsAnsweredExtractor)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.openai import OpenAI
from transformers import AutoTokenizer


# creds
os.environ['OPENAI_API_KEY'] = 'sk-...'
openai.api_key = os.environ['OPENAI_API_KEY']
input_dir_path = '/Users/marmik/test/'
cohere_api_key = '...'

# app title
st.set_page_config(page_title="💬📜 Chat w Papers")

# unique id session
if "id" not in st.session_state :
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None


def display_pdf(file):
    """opening file from file path"""

    st.markdown('### PDF preview')
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # embedding pdf in html
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    
    # displaying pdf
    st.markdown(pdf_display, unsafe_allow_html=True)


# welcome message
with st.chat_message("user"):
    st.write('Hello 👋')



with st.sidebar :
    st.title('💬📜 Chat w Papers')
    st.write("a RAG app powered by Meta's Llama 3-8b model")

    st.header(f'Add your documents')

    uploaded_file = st.file_uploader("choose your `.pdf` file ", type = 'pdf')

    if uploaded_file :
        try :
            with tempfile.TemporaryDirectory() as temp_dir :

                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                file_key = f'{session_id}-{uploaded_file.name}'

                st.write('indexing your document...')

                if file_key not in st.session_state.get('file_cache',{}):

                    if os.path.exists(temp_dir):

                        # loading the data
                        documents = SimpleDirectoryReader(input_dir = temp_dir,
                                                        required_exts=['.pdf'],
                                                        recursive = True).load_data(show_progress=True)

                    
                    else :
                        st.error('could not find the file you uploaded, please upload again...')
                        st.stop()

                    # docs = documents.load_data()

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

                    # create index
                    vector_index = VectorStoreIndex.from_documents(documents, 
                                                                transformations = [text_splitter, title_extractor, qa_extractor],
                                                                embed_model = embed_model,
                                                                show_progress = True)


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

                    llm = OpenAI(model="gpt-4o", temperature=0.5, max_tokens=256, tokenizer= tokenizer)

                    # llm = Ollama(model='llama3', request_timeout=120.0)

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


                    st.session_state.file_cache[file_key] = query_engine

                else :
                    query_engine = st.session_state.file_cache[file_key]

                # file is processed and ready to chat
                st.success('ready to chat')
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"error occured : {e}")
            st.stop()


col1, col2 = st.columns([6,1])

with col1 :
    st.header('chat w papers using llama-3')

with col2:
    st.button('clear ↺',on_click=reset_chat)


# initalize chat history 
if "messages" not in st.session_state :
    reset_chat()

# display chat messages from history on app rerun
for message in st.session_state.messages :
    print(message)
    with st.chat_message(message['role']):
        st.markdown(message['content'])



# accept user input
if prompt := st.chat_input('ask me something'):

    # display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    # add user message to chat history
    st.session_state.messages.append({'role':'user','content':prompt})

    # display assistant response in chat message container
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ''

        # simulate the stream of response with ms delay
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen :
            full_response += chunk
            message_placeholder.markdown(full_response + '|')

        message_placeholder.markdown(full_response)




    # add assistant response to chat history
    st.session_state.messages.append({'role':'assistant','content' : full_response})

