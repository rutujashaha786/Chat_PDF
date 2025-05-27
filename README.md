📚 Chat with PDF using Gemini
This project allows you to upload PDFs, process their text, and interactively ask questions about the content using Google Generative AI (gemini-flash). It uses Langchain for text processing, FAISS for vector storage, and Streamlit for the user interface.

Features
📄 Upload and Process PDF Files: Extracts text from PDFs and splits it into manageable chunks for querying.
🤖 Ask Questions: Interact with the text using natural language queries powered by Google Generative AI (gemini-flash).
🧠 Vector Store with FAISS: Efficient similarity search over processed text chunks.
🔥 Streamlit Web App: User-friendly interface to upload PDFs and ask questions directly.

🛠️ Installation
Clone the Repository:

git clone [https://github.com/rutujashaha786/Chat_PDF.git]
cd yourprojectname
Set Up a Virtual Environment:

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

pip install -r requirements.txt
Configure Environment Variables:

Create a .env file in the root directory with the following content:

GOOGLE_API_KEY=your_google_api_key
Replace your_google_api_key with your actual Google API key.

Run the Application:

streamlit run app.py
🧑‍💻 Usage
Upload one or more PDF files using the sidebar.
Click "Submit & Process" to extract text from the PDF and process it into chunks.
After processing is complete, you can ask questions about the PDF in the text input box, and app will respond based on the extracted content.