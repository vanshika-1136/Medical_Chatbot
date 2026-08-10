import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# SECURITY WARNING: Hardcoding your API key is okay for local testing, 
# but DO NOT upload this file to GitHub as-is, or your key will be stolen.
# If publishing, remove this line and use the .env setup instead.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm():
    # ChatGroq uses Groq's high-speed infrastructure and the highly capable Llama 3.3 70B model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    return llm


def main():
    st.title("Ask Chatbot!")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # React to user input
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        # Display user message in chat message container
        st.chat_message('user').markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Create the QA chain using our cleaner load_llm function
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Get the response
            response = qa_chain.invoke({'query': prompt})
            
            result = response["result"]
            source_documents = response["source_documents"]
            
            # Format the output to show sources clearly
            result_to_show = f"{result}\n\n**Source Docs:**\n```python\n{source_documents}\n```"
            
            # Display assistant response in chat message container
            st.chat_message('assistant').markdown(result_to_show)
            # Add assistant response to chat history
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()