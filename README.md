# 📚 Chat with PDF using Gemini

This project allows you to upload PDFs, process their text, and interactively ask questions about the content using Google Generative AI (`gemini-flash`). It uses Langchain for text processing, FAISS for vector storage, and Streamlit for the user interface.

## Features
- #### 📄 Upload and Process PDF Files: 
Extracts text from PDFs and splits it into manageable chunks for querying.
- #### 🤖 Ask Questions: 
Interact with the text using natural language queries powered by Google Generative AI (`gemini-flash`).
- #### 🧠 Vector Store with FAISS: 
Efficient similarity search over processed text chunks.
- #### 🔥 Streamlit Web App: 
User-friendly interface to upload PDFs and ask questions directly.

## 🛠️ Installation
1. #### Clone the Repository:

    ```bash
    git clone [https://github.com/rutujashaha786/Chat_PDF.git]
    cd yourprojectname
    ```
2. #### Set Up a Virtual Environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. #### Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. #### Configure Environment Variables:

    Create a `.env` file in the root directory with the following content:

    ```bash
    GOOGLE_API_KEY=your_google_api_key
    ```
    Replace `your_google_api_key` with your actual Google API key.

5. #### Run the Application:

    ```bash
    streamlit run app.py
    ```

### How to use
1. Upload one or more PDF files using the sidebar.
2. Click "Submit & Process" to extract text from the PDF and process it into chunks.
3. After processing is complete, you can ask questions about the PDF in the text input box, and app will respond based on the extracted content.