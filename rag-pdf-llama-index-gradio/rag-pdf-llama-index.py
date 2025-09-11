import time
import os
from huggingface_hub import login
from dotenv import load_dotenv
from utilities.document_handler import process_pdfs, split_docs, construct_nodes
from utilities.vector_handler import generate_nodes_embedding, vectorstore_nodes, \
    initiate_vector_store
from utilities.retriever import VectorDBRetriever
from utilities.database_handler import connect_db
from utilities.load_models import load_embedding_model, load_llm
from utilities.query_engine import construct_query_engine
from utilities.chat_handler import generate_response
from utilities.interface import gradio_interface
    
class rag_pdf_chatbot():
    """
    The RAG workflow class.

    """
    def __init__(self, hf_token,
                 model_path, 
                 embedding_path, 
                 chunk_size, 
                 chunk_overlap,
                 db_name,
                 db_type,
                 host,
                 user, 
                 password,
                 port,
                 table_name,
                 embed_dim
                 ):
        
        self.hf_token = hf_token
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # For connecting to the Postgres DB
        self.db_name = db_name
        self.db_type = db_type
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        
        # For creating a vector store
        self.table_name = table_name # the vector store name
        self.embed_dim = embed_dim
        
    def rag_workflow(self, pdfPath, query):
        """
        The RAG main process to generate responses to end user questions based on the information in the PDF file.

        Args:
            pdfPath (str): The PDF path
            query (str): The question from the end-user

        Returns:
            str: The LLM responses after cleaning 
        """
        # Login to access the private model
        login(token=hf_token)#inference-only token
        
        #------- Define the embedding model and the LLM -------
        embedding_model = load_embedding_model(self.embedding_path)
        llm = load_llm(model_path=self.model_path)
        
        #------- Ingestion Pipeline -------
        # 1. Load the PDF file and extract the text
        docs = process_pdfs(pdfPath)
        
        # 2. Split the documents into chunks
        doc_splits, doc_idxs = split_docs(docs, self.chunk_size, self.chunk_overlap)
        
        # 3. Construct nodes from text chunks
        nodes = construct_nodes(doc_splits, docs, doc_idxs)
        
        # 4. Generate embeddings from nodes using the embedding model
        embedded_nodes = generate_nodes_embedding(nodes, embedding_model)
        
        # 5.a. Connect to a Postgres DB 
        connect_db(self.db_name,
                   self.db_type,
                   self.host,
                   self.user,
                   self.password,
                   self.port)
        
        # 5.b. Create a vector store 
        vector_store = initiate_vector_store(self.db_name,
                                            self.host,
                                            self.user,
                                            self.password,
                                            self.port,
                                            self.table_name,
                                            self.embed_dim)
        
        # 5.c. Store the embedded nodes into a vector store
        vector_store = vectorstore_nodes(vector_store, embedded_nodes)
        
        #------- Retrieval pipeline -------
        # 6.a. Call the retriever
        retriever = VectorDBRetriever(vector_store, embedding_model, query_mode="default", similarity_top_k=5)
        
        # 6.b. Plug the retriever and the LLM into RetrieverQueryEngine
        query_engine = construct_query_engine(retriever, llm)
        
        # 7. Extract the response
        response = generate_response(query_engine, query)
        
        end_time = "\n\n--- Using %s, the time required to execute the application and provide a response : %.2f seconds ---"   %  (self.model_path, (time.time() - start_time))
        
        return response + end_time

if __name__ == "__main__":
    start_time = time.time()
    
    # Load environment variables from the .env file (if present)
    load_dotenv()
    
    # Read the environment variables 
    hf_token = os.getenv('HF_TOKEN')

    ## The LLM, embedding model, chunk size, and chunk overlap for document processing
    model_path = 'models/LLM/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
    embedding_path= 'kristanti/distiluse-base-matryoshka-french-labor-code'
    chunk_size = 1000
    chunk_overlap = 200
    
    ## The setting for connecting to a Postgres DB 
    db_name = 'postgres_vector_db' 
    db_type = 'postgres' 
    host = 'localhost'
    user = 'postgres'
    password = 'password'
    port = 5432
    
    ## The setting for creating a vector store in a Postgres DB
    table_name = 'vector_document'
    embed_dim = 512
    
    # Call the 
    rag_application = rag_pdf_chatbot(hf_token,
                                      model_path,
                                      embedding_path,
                                      chunk_size,
                                      chunk_overlap,
                                      db_name,
                                      db_type,
                                      host,
                                      user,
                                      password,
                                      port,
                                      table_name,
                                      embed_dim)
    
    # The Gradio interface
    gr_interface = gradio_interface(rag_application.rag_workflow)
    
    # Running the application
    gr_interface.launch()
    
    
    