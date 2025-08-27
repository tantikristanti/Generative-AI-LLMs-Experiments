from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

def process_pdfs(files):
    """
    Process Pdf files.

    Args:
        files (list): The list of the Pdf files

    Returns:
        documents(list): The list of LlamaIndex Document for storing a piece of text and associated metadata
    """
    
    documents = []
    
    file_names = [f.name for f in files]
    # Load the documents
    for path in file_names:
        loader = PyMuPDFReader()
        documents = loader.load(file_path=path)
        
    return documents

def split_docs(docs, chunk_size, chunk_overlap):
    """
    Split documents into chunks.

    Args:
        docs (list): The list of documents
        chunk_size (int): The maximum number of characters or tokens allowed in a single chunk
        chunk_overlap (int): The number of characters or tokens shared between consecutive chunks
        
    Returns:
        text_chunks (list): The list of splitted text 
        doc_idxs (list): The list of document indexes
    """
    
    # Create a sentence splitter to split the document
    text_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the document into text chunks
    text_chunks = []
    # Keep traces to the source document index for document metadata
    doc_idxs = []
    for idx, doc in enumerate(docs):
        chunks = text_parser.split_text(doc.text)
        text_chunks.extend(chunks)
        doc_idxs.extend([idx] * len(chunks))
    
    return text_chunks, doc_idxs

def construct_nodes(text_chunks, documents, doc_idxs):
    """
    Construct nodes from text chunks.

    Args:
        text_chunks (list): The list of splitted text 
        documents(list): The list of LlamaIndex Document for storing a piece of text and associated metadata
        doc_idxs (list): The list of document indexes

    Returns:
        nodes (list): The list of LlamaIndex nodes containing the text chunks and their corresponding source document metadata
    """

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        # each node contains the text and the text metadata
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata 
        nodes.append(node)
    
    return nodes
    