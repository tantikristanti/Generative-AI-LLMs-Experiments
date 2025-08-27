import time
import os
from dotenv import load_dotenv
from utilities.document_handler import process_pdfs, split_docs
from utilities.vector_handler import generate_embeddings, create_collection, create_vectorstore
from utilities.retriever import get_retriever
from utilities.chat_handler import generate_response
from utilities.interface import gradio_interface

class rag_pdf_chatbot():
    """
    The RAG workflow class.

    """
    def __init__(self, model_name, chunk_size, chunk_overlap):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def rag_workflow(self, pdfPath, question):
        """
        The RAG main process to generate responses to end user questions based on the information in the PDF file.

        Args:
            pdfPath (str): The PDF path
            question (str): The question from the end-user

        Returns:
            str: The LLM responses after cleaning 
        """
        
        # 1. Load the PDF file and extract the text
        docs = process_pdfs(pdfPath)
        
        # 2. Split the documents into chunks
        doc_splits = split_docs(docs, self.chunk_size, self.chunk_overlap)
        
        # 3. Create embeddings
        embeddings = generate_embeddings(doc_splits, self.model_name)
        
        # 4.a. Create a collection of document chunks and their corresponding embeddings
        collection_name="rag-chroma"
        create_collection(doc_splits, embeddings, collection_name)
        
        # 4.b. Save the collection to a vector store
        # Make sure to use the same collection name  
        vectorstore = create_vectorstore(doc_splits, collection_name, self.model_name)
        
        # 5. Retrieve relevant context from the Chroma collection based on the question
        retriever = get_retriever(vectorstore)

        # 6. Format the input prompt and generate the response
        response = generate_response(self.model_name, question, retriever)
        
        end_time = "\n\n--- Time required to execute the application and provide a response : %.2f seconds ---" % (time.time() - start_time)
        
        return response + end_time

if __name__ == "__main__":
    start_time = time.time()
    
    # Load environment variables from the .env file (if present)
    load_dotenv()

    # Access the model name from the environment variables 
    model_name = os.getenv('MODEL_NAME')
    chunk_size = int(os.getenv('CHUNK_SIZE'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP'))
    
    rag_application = rag_pdf_chatbot(model_name, chunk_size, chunk_overlap)
    
    # The Gradio interface
    gr_interface = gradio_interface(rag_application.rag_workflow)
    
    # Running the application
    gr_interface.launch()
    