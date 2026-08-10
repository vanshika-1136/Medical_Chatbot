import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
# Step 1: Setup Groq API Key

os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"

def load_llm():
    # ChatGroq uses Groq's high-speed infrastructure and avoids HF API provider bugs
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    return llm

# Step 2: Prompt Setup
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Run Query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])