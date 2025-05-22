import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

vector_store = None
retrieval_chain = None
api_key_set = False

def set_api_key(user_key):
    global api_key_set
    if not user_key.strip():
        return "Please enter a valid API key."
    os.environ["GOOGLE_API_KEY"] = user_key.strip()
    api_key_set = True
    return "API key set. You can now upload a PDF."

def process_pdf(file_obj):
    global vector_store, retrieval_chain

    if not api_key_set:
        return "Set your API key first."

    reader = PdfReader(file_obj.name)
    text = ''.join(page.extract_text() or '' for page in reader.pages).strip()
    if not text:
        return "No text found in PDF."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using the context. Say 'I don't know' if not in context."),
        ("human", "Context: {context}\n\nQuestion: {input}")
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), doc_chain)

    return "PDF processed. You can now ask a question."

def ask_question(question):
    if not retrieval_chain:
        return "RAG pipeline not set up. Please process a PDF first."
    result = retrieval_chain.invoke({"input": question})
    return result.get("answer", "No answer found.")

with gr.Blocks() as demo:
    gr.Markdown("## AI RAG Chatbot")
    
    api_key = gr.Textbox(label="Enter your Google API Key", type="password", placeholder="Paste your Google Gemini API key here")
    key_btn = gr.Button("Set API Key")
    key_status = gr.Textbox(label="API Key Status", interactive=False)

    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_btn = gr.Button("Process PDF")
    status = gr.Textbox(label="Processing Status", interactive=False)

    question = gr.Textbox(label="Ask a Question")
    answer = gr.Textbox(label="Answer", interactive=False)

    key_btn.click(set_api_key, inputs=[api_key], outputs=[key_status])
    process_btn.click(process_pdf, inputs=[pdf_input], outputs=[status])
    question.submit(ask_question, inputs=[question], outputs=[answer])

demo.launch(share=True)
