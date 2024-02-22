from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    pdf_path = (
        "C:\\Users\\chho\\dev\\repos\\learning-langchain\\project-4\\2210.03629.pdf"
    )

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    docs = text_splitter.split_documents(documents=documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    # Initializing the vectore store object with FAISS for in memory vector database
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    # We can save the vectorstore to the disk
    vectorstore.save_local(
        "C:\\Users\\chho\\dev\\repos\\learning-langchain\\project-4\\faiss_index_react"
    )

    # Now we are loading the vectorstore from the disk
    loaded_vectorstore = FAISS.load_local(
        "C:\\Users\\chho\\dev\\repos\\learning-langchain\\project-4\\faiss_index_react",
        embeddings=embeddings,
    )

    # RetrievalQA is a chain object that is used to chain the retriever and the language model
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        chain_type="stuff",
        retriever=loaded_vectorstore.as_retriever(),
    )
    # Now we can ask questions to the model
    res = qa.run("Give me the gist of ReAct in 3 sentences")
    # And we should get the response with the context of our pdf document but answered by GPT-3
    print(res)
