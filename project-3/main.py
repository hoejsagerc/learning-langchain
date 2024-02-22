"""
as of running through this tutorial i get an import error when running automatic versioning.
therefore i have downgraded some of the langchain packages

- langchain==0.1.7 => langchain==0.1.6
- langchain-community==0.0.20 => langchain-community==0.0.19
"""

from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from langchain_pinecone import Pinecone


from dotenv import load_dotenv
from pinecone import Pinecone as pine
import os


load_dotenv()

# Using the pinecone-client library to connect to the Pinecone API
pc = pine(api_key=os.environ["PINECONE_API_KEY"])


if __name__ == "__main__":
    # Creating the laoder object
    loader = TextLoader(
        file_path="C:\\Users\\chho\\dev\\repos\learning-langchain\\project-3\\mediumblogs\\mediumblog1.txt",
        encoding="utf-8",
    )

    # Laoding the text file from the local file and encoding as utf-9
    document = loader.load()

    # If we see that our prompt is not reaponding as expected, then the two parameters below can be adjusted
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    # Using the OpenAIEmbeddings to convert the text into embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    # loading the embeddings into the Pinecone index => once this is run then you should see the embeddings in the pinecone dashboard
    # docsearch is a vector store object
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    # RetrievalQA is a chain object that is used to chain the retriever and the language model
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    query = "What is a vector DB? give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
