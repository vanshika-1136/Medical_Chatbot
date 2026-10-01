import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource 
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


def load_llm():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Please add it to your .env file."
        )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.3
    )

    return llm


def main():

    st.title("Ask Chatbot!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User input
    prompt = st.chat_input("Pass your prompt here")

    if prompt:

        # Display user message
        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.

        If you don't know the answer, just say that you don't know.
        Don't try to make up an answer.
        Don't provide anything outside of the given context.

        Context: {context}

        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:

            # Load FAISS vector database
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            # Create Retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt(
                        CUSTOM_PROMPT_TEMPLATE
                    )
                }
            )

            # Get response
            response = qa_chain.invoke({
                "query": prompt
            })

            result = response["result"]
            source_documents = response["source_documents"]

            # Display answer
            st.chat_message("assistant").markdown(result)

            # Display sources
            with st.expander("View Source Documents"):
                for i, doc in enumerate(source_documents, start=1):

                    st.markdown(f"### Source {i}")

                    st.write(doc.page_content)

                    if hasattr(doc, "metadata"):
                        st.caption(doc.metadata)

            # Save response in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()



# import os
# from dotenv import load_dotenv
# import streamlit as st

# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_classic.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # SECURITY WARNING: Hardcoding your API key is okay for local testing, 
# # but DO NOT upload this file to GitHub as-is, or your key will be stolen.
# # If publishing, remove this line and use the .env setup instead.
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # from dotenv import load_dotenv, find_dotenv
# # load_dotenv(find_dotenv())

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm():
#     # ChatGroq uses Groq's high-speed infrastructure and the highly capable Llama 3.3 70B model
#     llm = ChatGroq(
#         model_name="openai/gpt-oss-120b",
#         temperature=0.3
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     # Initialize chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     # React to user input
#     prompt = st.chat_input("Pass your prompt here")

#     if prompt:
#         # Display user message in chat message container
#         st.chat_message('user').markdown(prompt)
#         # Add user message to chat history
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer user's question.
#         If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#         Dont provide anything out of the given context

#         Context: {context}
#         Question: {question}

#         Start the answer directly. No small talk please.
#         """
        
#         try: 
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")
#                 return

#             # Create the QA chain using our cleaner load_llm function
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             # Get the response
#             response = qa_chain.invoke({'query': prompt})
            
#             result = response["result"]
#             source_documents = response["source_documents"]
            
#             # Format the output to show sources clearly
#             result_to_show = f"{result}\n\n**Source Docs:**\n```python\n{source_documents}\n```"
            
#             # Display assistant response in chat message container
#             st.chat_message('assistant').markdown(result_to_show)
#             # Add assistant response to chat history
#             st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()