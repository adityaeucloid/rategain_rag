import os
import glob
import logging
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

PROGRESS_FILE = "embedding_progress.json"

# read all pdf files from Documents folder and return text
def get_pdf_text_from_documents():
    documents_folder = os.path.join(os.getcwd(), "Documents")
    pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files in Documents folder.")
    text = ""
    for pdf_path in pdf_files:
        logging.info(f"Reading: {pdf_path}")
        with open(pdf_path, "rb") as pdf:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                logging.debug(f"Extracted page {page_num+1} from {pdf_path}, {len(page_text) if page_text else 0} chars.")
    logging.info(f"Total extracted text length: {len(text)} characters.")
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    logging.info(f"Split text into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        logging.debug(f"Chunk {i+1}: {len(chunk)} characters.")
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    logging.info(f"Embedding {len(chunks)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07")  # type: ignore
    # Optionally, embed in batches and log progress
    try:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        logging.info("Embedding completed successfully.")
        vector_store.save_local("faiss_index")
        logging.info("FAISS index saved to 'faiss_index'.")
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        raise


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro",
                                   client=genai,
                                   temperature=0.1,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your Documents PDFs."}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def build_vector_store_incrementally():
    documents_folder = os.path.join(os.getcwd(), "Documents")
    pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files in Documents folder.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")  # type: ignore

    # Load progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()

    # Try to load existing index, else create new
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logging.info("Loaded existing FAISS index.")
    else:
        vector_store = None

    for pdf_path in pdf_files:
        if pdf_path in processed_files:
            logging.info(f"Skipping already processed: {pdf_path}")
            continue
        logging.info(f"Processing: {pdf_path}")
        with open(pdf_path, "rb") as pdf:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text:
            logging.warning(f"No text found in {pdf_path}, skipping.")
            processed_files.add(pdf_path)
            continue
        chunks = get_text_chunks(text)
        logging.info(f"Split {pdf_path} into {len(chunks)} chunks.")
        try:
            if vector_store is None:
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            else:
                vector_store.add_texts(chunks)
            vector_store.save_local("faiss_index")
            logging.info(f"Added {pdf_path} to FAISS index and saved.")
            processed_files.add(pdf_path)
            with open(PROGRESS_FILE, "w") as f:
                json.dump(list(processed_files), f)
        except Exception as e:
            logging.error(f"Error embedding {pdf_path}: {e}")
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                logging.error("Quota exhausted. Stopping further processing.")
                break


def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Build knowledge base if not already present or not complete
    if not os.path.exists("faiss_index") or True:
        with st.spinner("Building knowledge base from Documents PDFs (incremental)..."):
            build_vector_store_incrementally()
            st.success("Knowledge base built/updated from Documents folder.")

    st.title("Chat with PDF files🤖")
    st.write("Knowledge base loaded from Documents folder.")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your Documents PDFs."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()