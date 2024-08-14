# from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def ingest_docs():
    loader = DirectoryLoader(
        path="project-5_docs_helper\langchain-docs",
        show_progress=True,
    )
    raw_docs = loader.load()
    print(f"loaded {len(raw_docs)} documents from langchain-docs/")


if __name__ == "__main__":
    ingest_docs()