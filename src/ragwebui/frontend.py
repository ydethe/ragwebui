from typing import List, Tuple

import gradio as gr

from .RAGRequestor import RAGRequestor


def start_frontend(rag_requestor: RAGRequestor):
    def handle_question(message: str, chat_history: List[dict]) -> Tuple[str, List[str], str]:
        rag_request = rag_requestor.rag_with_anchored_sources(message)
        answer, html_sources = rag_request.to_html()

        chat_history = [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": message},
        ] + chat_history

        return "", chat_history, html_sources

    with gr.Blocks() as iface:
        gr.Markdown("# Chatdoc")
        msg = gr.Textbox(label="Question")
        chatbot = gr.Chatbot(type="messages", label="Historique")
        sources = gr.HTML(label="Sources")
        # clear = gr.ClearButton([msg, chatbot])

        msg.submit(
            fn=handle_question,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, sources],
        )

    iface.launch(server_name="0.0.0.0", pwa=True)
