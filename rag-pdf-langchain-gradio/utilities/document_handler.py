from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

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

def split_docs(docs, embedding_model):
    """
    Split documents into chunks.

    Args:
        docs (list): The list of documents
        chunk_size (int): The maximum number of characters or tokens allowed in a single chunk
        chunk_overlap (int): The number of characters or tokens shared between consecutive chunks
        
    Returns:
        chunks (list): The list of splitted documents
    """
    
    # Check the maximum size of the embedding model
    max_length_size = SentenceTransformer(embedding_model).max_seq_length
    
    # Split the documents into smaller chunks. The size of each chunk should be less than the maximum size of the embedding model 
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(embedding_model),
        chunk_size=max_length_size,
        chunk_overlap=int(max_length_size / 10),
        add_start_index=True, # set to True to include chunk's start index in metadata `True`, includes chunk's start index in metadata
        strip_whitespace=True, # set to True to strip whitespace from the start and end of every document
    )
    
    docs_processed = []
    for doc in docs:
        docs_processed += text_splitter.split_documents([doc])
        
    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    
    return docs_processed_unique