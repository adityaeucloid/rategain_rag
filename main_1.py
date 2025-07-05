import os
import glob
import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor
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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

PROGRESS_FILE = "embedding_progress.json"

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

def tag_metric_chunks(chunks):
    """Tag chunks containing metric-related content (e.g., revenue, EBITDA)."""
    metric_keywords = ['revenue', 'ebitda', 'margin', 'cost', 'budget', 'forecast']
    tagged_chunks = []
    for chunk in chunks:
        is_metric = any(keyword in chunk.lower() for keyword in metric_keywords)
        tagged_chunks.append({
            'text': chunk,
            'is_metric': is_metric
        })
    return tagged_chunks

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    logging.info(f"Split text into {len(chunks)} chunks.")
    tagged_chunks = tag_metric_chunks(chunks)
    for i, chunk in enumerate(tagged_chunks):
        logging.debug(f"Chunk {i+1}: {len(chunk['text'])} chars, Metric: {chunk['is_metric']}")
    return tagged_chunks

def parallel_similarity_search(new_db, query, chunks, max_workers=4):
    """Perform similarity search in parallel across document chunks."""
    def search_chunk(chunk):
        docs = new_db.similarity_search_with_score(chunk['text'], k=5)
        return [(doc, score, chunk['is_metric']) for doc, score in docs]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(search_chunk, chunk): chunk for chunk in chunks}
        for future in future_to_chunk:
            try:
                results.extend(future.result())
            except Exception as e:
                logging.error(f"Error in parallel search: {e}")
    
    # Sort by similarity score (lower is better for FAISS)
    results.sort(key=lambda x: x[1])
    
    # Prioritize metric-related chunks for metric queries
    if any(keyword in query.lower() for keyword in ['revenue', 'ebitda', 'margin', 'cost', 'budget', 'forecast']):
        results = sorted(results, key=lambda x: (not x[2], x[1]))  # Prioritize metric chunks
    
    # Return top-k documents
    top_k = 5
    return [doc for doc, score, _ in results[:top_k]]

def get_vector_store(tagged_chunks):
    logging.info(f"Embedding {len(tagged_chunks)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    try:
        texts = [chunk['text'] for chunk in tagged_chunks]
        vector_store = FAISS.from_texts(texts, embedding=embeddings)
        logging.info("Embedding completed successfully.")
        vector_store.save_local("faiss_index")
        logging.info("FAISS index saved to 'faiss_index'.")
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        raise
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an expert financial analyst with detailed knowledge of RateGain Travel Technologies Limited, a global provider of SaaS solutions for the travel and hospitality industry. RateGain offers smart technology to optimize revenue, distribution, and marketing for travel companies. Its key business units include:
    - **DaaS (Data as a Service)**: Includes Travel BI (business intelligence for airlines, OTAs, and travel companies) and Hospi BI (business intelligence for hospitality). These products provide data-driven insights for revenue optimization.
    - **Distribution**: Includes DHISCO and RezGain, which focus on enterprise connectivity and distribution solutions for hotels and travel intermediaries, excluding IHG-specific operations.
    - **MarTech**: Includes Adara, BCV (Brand Content Velocity), and MHS (Marketing Hospitality Solutions), focusing on marketing technology and social media solutions for travel brands.
    
    Financial Context (based on FY25 H1 and September 2024 data):
    - FY25 H1 gross revenue: $62.8 million (including Adara and Knowland), up 14% year-over-year but below the $64.9 million budget.
    - H1 EBITDA: $12.27 million, slightly below the $12.6 million budget, with a margin of 19.5%.
    - September 2024 gross revenue: $11.59 million, with a gross margin of 7.2% and EBITDA of $2.89 million.
    - Challenges include lower distribution performance due to AWS costs and a decline in new bookings.
    - Key clients include H World, Jarnah Hotels, and others listed in the UNO revenue breakdown.

    Instructions:
    - For questions about financial metrics (e.g., revenue, EBITDA, margins), extract and summarize the data in a structured format (e.g., bullet points or tables).
    - For non-metric questions, provide detailed answers based on the context, incorporating RateGain’s business context where relevant.
    - If the answer is not available in the provided context, state: "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", client=genai, temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your Documents PDFs."}]

def user_input(user_question, tagged_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform parallel similarity search
    docs = parallel_similarity_search(new_db, user_question, tagged_chunks)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response['output_text']

def build_vector_store_incrementally():
    documents_folder = os.path.join(os.getcwd(), "Documents")
    pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files in Documents folder.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()

    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        logging.info("Loaded existing FAISS index.")
    else:
        vector_store = None

    tagged_chunks_all = []
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
        tagged_chunks = get_text_chunks(text)
        tagged_chunks_all.extend(tagged_chunks)
        try:
            if vector_store is None:
                vector_store = get_vector_store(tagged_chunks)
            else:
                texts = [chunk['text'] for chunk in tagged_chunks]
                vector_store.add_texts(texts)
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
    return tagged_chunks_all

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")
    
    # Build knowledge base incrementally
    if not os.path.exists("faiss_index"):
        with st.spinner("Building knowledge base from Documents PDFs (incremental)..."):
            tagged_chunks = build_vector_store_incrementally()
            st.success("Knowledge base built/updated from Documents folder.")
    else:
        with open(PROGRESS_FILE, "r") as f:
            processed_files = set(json.load(f))
        text = get_pdf_text_from_documents()
        tagged_chunks = get_text_chunks(text)

    st.title("Chat with PDF files using Gemini🤖")
    st.write("Knowledge base loaded from Documents folder.")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your Documents PDFs."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt, tagged_chunks)
                placeholder = st.empty()
                full_response = response
                placeholder.markdown(full_response)
        if response:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()