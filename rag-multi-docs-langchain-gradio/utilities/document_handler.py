from langchain_community.document_loaders import TextLoader, \
    PDFPlumberLoader, Docx2txtLoader, CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

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