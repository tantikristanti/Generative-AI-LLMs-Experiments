import gradio as gr

def gradio_interface(application):
    """
    The Gradio interface. 

    """
    
    # Define the Gradio interface
    interface = gr.Interface(fn=application,
                inputs=[gr.File(file_count='multiple'), gr.Textbox(label="Enter questions based on the information in the given PDF file.")],
                outputs=[gr.Textbox(label="The agent's answer:")],
                title="PDF Document Query",
                description="Enter a question to query the document.")
    
    return interface