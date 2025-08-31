from langchain_community.document_loaders import TextLoader, \
    PDFPlumberLoader, Docx2txtLoader, CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document_from_files(files) -> list[Document]:
    """
    Process files (*.txt, *.pdf, *.docx).

    Args:
        files (list): The list of the files

    Returns:
        list[Document]: The list of LangChain Document for storing the content and the associated metadata of the document
    """
    
    loader = None
    documents = []
        
    file_names = [f.name for f in files]
    for path in file_names:
        suffix = "." + path.split(".")[-1]
        if suffix == ".pdf":
            loader = PDFPlumberLoader(path)
        elif suffix == ".txt":
            loader = TextLoader(path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(path)
        elif suffix == ".csv":
            loader = CSVLoader(path)
        else:
            raise ValueError("Unsupported file type")
    
        # Load and collect the document
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