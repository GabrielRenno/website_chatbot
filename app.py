import streamlit as st
from dotenv import load_dotenv
import os
import openai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

def run_chatbot(website, question):
    # Load .env file
    load_dotenv()
    # Set your OpenAI API key here
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load documents
    # Url as string
    loader = WebBaseLoader(website)
    docs = loader.load()

    # Split text into chunks
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    splits = r_splitter.split_text(docs[0].page_content)

    # Embed chunks
    embedding = OpenAIEmbeddings()
    # Create ChatOpenAI instance
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    vectordb = FAISS.from_texts(texts=splits, embedding=embedding)
    # Build prompt
    template = """ You are the chatbot for this company. You plotely answer the questions of the customers giving a lot of details based on what you find in the context. Do not say anything that is not in the website
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    # Provide the "company" input key when calling the qa_chain function
    result = qa_chain({"query": question})

    return result["result"]

def main():
    st.title("Chatbot App")
    
    website = st.text_input("Enter your website")
    question = st.text_input("Enter your question")
    
    if st.button("Submit"):
        response = run_chatbot(website, question)
        st.write("Chatbot's response:")
        st.write(response)

if __name__ == "__main__":
    main()
