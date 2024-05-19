import streamlit as st
import random
import time

# app title
st.set_page_config(page_title="💬📜 Chat w Papers")

# initalize chat history 
if "messages" not in st.session_state :
    st.session_state.messages = []

# welcome message
with st.chat_message("user"):
    st.write('Hello 👋')

# display a chat input widget and show the user's input
# prompt = st.chat_input("Ask something")

# if prompt :
#     st.write(f'User has sent the following prompt : {prompt}')


# display chat messages from history on app rerun
for message in st.session_state.messages :
    print(message)
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# react to user input
if prompt := st.chat_input('ask me something'):

    #display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    # add user message to chat history
    st.session_state.messages.append({'role':'user','content':prompt})

# add model's response
response = f"Echo : {prompt}"

#display assistant response in chat message container
with st.chat_message('assistant'):
    st.markdown(response)

# add assistant response to chat history
st.session_state.messages.append({'role':'assistant','content':response})

with st.sidebar :
    st.title('💬📜 Chat w Papers')
    st.write("a RAG app powered by Meta's Llama 3-8b model")

