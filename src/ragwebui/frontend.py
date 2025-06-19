import gradio as gr

from .RAGRequestor import RAGRequestor


def start_frontend(rag_requestor: RAGRequestor):
    with gr.Blocks() as iface:
        gr.Markdown("# Chatdoc")
        msg = gr.Textbox(label="Question")
        chatbot = gr.Chatbot(type="messages", label="Historique")
        sources = gr.HTML(label="Sources")
        # clear = gr.ClearButton([msg, chatbot])

        msg.submit(
            fn=rag_requestor.rag_with_anchored_sources,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, sources],
        )

    iface.launch(server_name="0.0.0.0", pwa=True)
