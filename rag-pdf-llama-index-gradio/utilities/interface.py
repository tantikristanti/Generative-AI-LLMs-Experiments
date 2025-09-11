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

def gradio_interface_theme_1(application):
    """
    The Gradio interface.

    Args:
        pdfPath (str): The PDF path
        question (str): The question from the end-user

    Returns:
        str: The LLM responses after cleaning 
    """
    
    # Define the Gradio interface with theme
    with gr.Blocks(
        fill_height=True,
        fill_width=True,
        analytics_enabled=False,
        theme=gr.themes.Ocean(
            radius_size="md",
            #https://fonts.google.com/
            font=gr.themes.GoogleFont("Quicksand", weights=(100, 300)),
            font_mono=gr.themes.GoogleFont("Karla", weights=(100, 300)),
        ),
    ) as interface:
        gr.Interface(
                    # call the function in fn  
                    fn=application,
                    # define inputs to the function in fn; make sure that the list have the same number of called function parameters
                    inputs=[gr.File(file_count='multiple'), gr.Textbox(label="Enter questions based on the information in the given PDF file.")], 
                    # define outputs from the function in fn; make sure that the list have the same number of called function outputs
                    outputs=[gr.Textbox(label="The agent's answer:")],
                    title="PDF Document Query",
                    description="Enter a question to query the document."
                    )
    return interface

def gradio_interface_theme_2(application):
    """
    The Gradio interface.

    Args:
        pdfPath (str): The PDF path
        question (str): The question from the end-user

    Returns:
        str: The LLM responses after cleaning 
    """
    
    # Define the Gradio interface with theme
    with gr.Blocks(
        fill_height=True,
        fill_width=True,
        analytics_enabled=False,
        theme=gr.themes.Glass(
            radius_size="md",
            #https://fonts.google.com/
            font=gr.themes.GoogleFont("Lato", weights=(100, 300)),
            font_mono=gr.themes.GoogleFont("IBM Plex Mono", weights=(100, 300)),
        ),
    ) as interface:
        gr.Interface(
                    # call the function in fn  
                    fn=application,
                    # define inputs to the function in fn; make sure that the list have the same number of called function parameters
                    inputs=[gr.File(file_count='directory'), gr.Textbox(label="Enter questions based on the information in the given PDF file.")], 
                    # define outputs from the function in fn; make sure that the list have the same number of called function outputs
                    outputs=[gr.Textbox(label="The agent's answer:")],
                    title="PDFs Document Query",
                    description="Enter a question to query the document."
                    )
    return interface