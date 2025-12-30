import os
import shutil
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai 
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

def submit():
    st.session_state.user_text = st.session_state.widget
    st.session_state.widget = ""

def get_pdf_text(pdf_docs):
    text = ""
    scanned_pdfs = []
    read_failed_pdfs = []

    for pdf in pdf_docs:
        pdf_text = ""
        try:
            pdf_reader = PdfReader(pdf) 
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
        except PdfReadError as e:
            read_failed_pdfs.append(pdf.name)
            st.error(f"❌ Failed to read {pdf.name}: {str(e)}")
            continue
        except Exception as e:
            read_failed_pdfs.append(pdf.name)
            st.error(f"❌ Unexpected error reading {pdf.name}: {str(e)}")
            continue

        if not pdf_text.strip():
            scanned_pdfs.append(pdf.name)
            st.error(f"❌ No extractable text found in {pdf.name}. Please upload a text-based PDF.")
        else:
            text += pdf_text
    return text, scanned_pdfs, read_failed_pdfs

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the provided context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature = 0.3)
    prompt = PromptTemplate(input_variables=["context", "question"], template = prompt_template)
    stuff_chain = create_stuff_documents_chain(model, prompt)
    return stuff_chain

def process_user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("❌ Please upload and process PDFs first before asking questions.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)

        chain = get_conversational_chain()
        response = chain.invoke({"context": docs, "question": user_question})
        st.write("***Question***:", user_question)
        st.write("***Reply***:", response)
        st.session_state.user_text = ""
    except Exception as e:
        if "RATE_LIMIT_EXCEEDED" in str(e):
            st.error("❌ Our system is receiving too many requests at the moment. Please wait a minute and try again.")
        st.error(f"❌ Error processing your question: {str(e)}")
    

def main():
    st.set_page_config(page_title="Chat With PDF", page_icon=":books:")
    st.header("Chat with PDF using Gemini :books:")

    st.text_input("Ask a question from your PDF Files:", key="widget", on_change=submit)
    user_question = st.session_state.user_text

    if user_question:
        process_user_input(user_question)

    with st.sidebar:
        st.title("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on 'Submit & Process' Button", accept_multiple_files=True, type="pdf")
        
        if st.button("Submit & Process"):
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF file to process.")
            else:
                with st.spinner("Processing..."):
                    raw_text, scanned_pdfs, read_failed_pdfs = get_pdf_text(pdf_docs)
                    if not raw_text.strip() and scanned_pdfs:
                        return

                    if not raw_text.strip() and read_failed_pdfs:
                        return

                    try:
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("❌ No text chunks generated. Please check the PDF content.")
                            return
                        get_vector_store(text_chunks)
                        st.success("✅ Processing complete! You can now ask your question.")
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        return                    


if __name__ == "__main__":
    main()