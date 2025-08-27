from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdfs(files):
    """
    Process Pdf files.

    Args:
        files (list): The list of the Pdf files

    Returns:
        documents (list): The list of LangChain Document for storing a piece of text and associated metadata
    """
    
    documents = []
    
    file_names = [f.name for f in files]
    # Load the documents
    for path in file_names:
        loader = PDFPlumberLoader(path)
        documents.extend(loader.load())
    
    return documents

def split_docs(docs, chunk_size, chunk_overlap):
    """
    Split documents into chunks.

    Args:
        docs (list): The list of documents
        chunk_size (int): The maximum number of characters or tokens allowed in a single chunk
        chunk_overlap (int): The number of characters or tokens shared between consecutive chunks
        
    Returns:
        chunks (list): The list of splitted documents
    """
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    
    return chunks