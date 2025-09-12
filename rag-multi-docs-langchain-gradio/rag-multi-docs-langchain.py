import time
import os
from dotenv import load_dotenv 
from huggingface_hub import login
from utilities.document_handler import load_document_from_files, split_docs
from utilities.vector_handler import generate_embeddings, create_collection, create_vector_store
from utilities.retriever import get_retriever
from utilities.chat_handler import generate_response
from utilities.interface import gradio_interface

class rag_pdf_chatbot():
    """
    The RAG workflow class.

    """
    def __init__(self, hf_token, model_name, embedding_name, chroma_path):
        self.hf_token = hf_token
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.chroma_path = chroma_path
    
    def rag_workflow(self, pdfPath, query):
        """
        The RAG main process to generate responses to end user questions based on the information in the PDF file.

        Args:
            pdfPath (str): The PDF path
            query (str): The question from user

        Returns:
            str: The LLM responses after cleaning 
        """
        
        # Login to access the private model
        #login(token=hf_token)#inference-only token
        
        # 1. Load the PDF file and extract the text
        docs = load_document_from_files(pdfPath)
        
        # 2. Split the documents into chunks
        doc_splits = split_docs(docs, embedding_model=self.embedding_name)
        
        # 3. Create embeddings using the embedding model 
        embeddings = generate_embeddings(doc_splits, embedding_model=self.embedding_name)
        
        # 4. Create a collection and store in a vector store if the collection does not exist yet
        create_collection(doc_splits, embeddings)
        vectorstore = create_vector_store(doc_splits, self.embedding_name)
        
        # 5. Retrieve relevant context from the Chroma collection based on the question
        retriever = get_retriever(vectorstore)

        # 6. Format the input prompt and generate the response
        response = generate_response(self.model_name, query, retriever)
        
        end_time = "\n\n--- Using %s, the time required to execute the application and provide a response : %.2f seconds ---"   %  (self.model_name, (time.time() - start_time))
        return response + end_time

if __name__ == "__main__":
    start_time = time.time()
    
    # Load environment variables from the .env file (if presents)
    load_dotenv()

    # Access the model name from the environment variables 
    hf_token = os.getenv('HF_TOKEN')
    model_name = 'llama3.2' 
    embedding_name = "BAAI/bge-base-en-v1.5" #"kristanti/distiluse-base-matryoshka-french-labor-code"
    chroma_path = './chroma_db'
    
    rag_application = rag_pdf_chatbot(hf_token, model_name, embedding_name, chroma_path)
    
    # The Gradio interface
    gr_interface = gradio_interface(rag_application.rag_workflow)
    
    # Running the application
    gr_interface.launch()
    