import os
import numpy as np
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
import nest_asyncio
import PyPDF2
from pathlib import Path
import logging
import json
import time
import hashlib

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)

WORKING_DIR = "./chickens"

# Only recreate the working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
else:
    # Clean up kv store encoding issues if directory exists
    cleanup_kv_store_encoding()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    combined_prompt += f"user: {prompt}"

    # Call the Gemini model
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(
        combined_prompt,
        generation_config=GenerationConfig(max_output_tokens=100000, temperature=0.1),
    )

    return response.text

async def embedding_func(texts: list[str]) -> np.ndarray:
    # Use text-embedding-004 model for embeddings
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = np.array(result['embedding'])
    return embeddings

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # text-embedding-004 produces 768-dimensional embeddings
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

def get_file_hash(file_path):
    """Generate a hash for the file to track if it has changed."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return None

def load_processing_status():
    """Load the current processing status from LightRAG's kv store."""
    kv_store_path = Path(WORKING_DIR) / "kv_store_doc_status.json"
    if kv_store_path.exists():
        try:
            with open(kv_store_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading processing status: {e}")
            # Try to read with different encoding if UTF-8 fails
            try:
                with open(kv_store_path, 'r', encoding='latin-1') as f:
                    return json.load(f)
            except Exception as e2:
                logger.error(f"Error loading processing status with latin-1: {e2}")
    return {}

def is_document_processed(file_path, processing_status):
    """Check if a document has already been processed."""
    file_hash = get_file_hash(file_path)
    if not file_hash:
        return False
    
    # Check if any document in the status has this file path or hash
    for doc_id, doc_info in processing_status.items():
        if (doc_info.get('file_path') == str(file_path) or 
            doc_info.get('file_hash') == file_hash):
            # Consider both 'processed' and 'processing' as already handled
            if doc_info.get('status') in ['processed', 'processing']:
                logger.info(f"Document {file_path} found in status: {doc_info.get('status')}")
                return True
    return False

def update_document_status(file_path, doc_id, status="processed"):
    """Update the document status in the kv store."""
    kv_store_path = Path(WORKING_DIR) / "kv_store_doc_status.json"
    if kv_store_path.exists():
        try:
            with open(kv_store_path, 'r', encoding='utf-8') as f:
                doc_status = json.load(f)
        except Exception as e:
            logger.error(f"Error reading kv store: {e}")
            try:
                with open(kv_store_path, 'r', encoding='latin-1') as f:
                    doc_status = json.load(f)
            except Exception as e2:
                logger.error(f"Error reading kv store with latin-1: {e2}")
                doc_status = {}
    else:
        doc_status = {}
    
    # Update the document info
    if doc_id in doc_status:
        doc_status[doc_id]['file_path'] = str(file_path)
        doc_status[doc_id]['file_hash'] = get_file_hash(file_path)
        doc_status[doc_id]['status'] = status
        doc_status[doc_id]['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    # Save back to file with UTF-8 encoding
    try:
        with open(kv_store_path, 'w', encoding='utf-8') as f:
            json.dump(doc_status, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated status for {file_path}")
    except Exception as e:
        logger.error(f"Error updating kv store: {e}")

async def process_single_document(rag, pdf_file):
    """Process a single PDF document asynchronously."""
    logger.info(f"Starting to process: {pdf_file}")
    try:
        text = extract_text_from_pdf(pdf_file)
        logger.info(f"Text extraction complete for {pdf_file}. Text length: {len(text)}")
        
        if text.strip():
            logger.info(f"Inserting document into RAG: {pdf_file}")
            # Use async insert without timeout
            try:
                result = await rag.ainsert(text)
                logger.info(f"Successfully inserted: {pdf_file}")
                
                # Extract doc_id from result if possible, or generate one
                doc_id = None
                if hasattr(result, 'doc_id'):
                    doc_id = result.doc_id
                elif isinstance(result, dict) and 'doc_id' in result:
                    doc_id = result['doc_id']
                else:
                    # Generate a doc_id based on file path and content
                    doc_id = f"doc-{hashlib.md5(f'{pdf_file}{len(text)}'.encode()).hexdigest()[:16]}"
                
                # Update the document status with file path
                update_document_status(pdf_file, doc_id, "processed")
                
                return {"file": pdf_file, "status": "success", "chunks": len(text.split()), "doc_id": doc_id}
            except Exception as e:
                logger.error(f"Error inserting {pdf_file}: {e}")
                return {"file": pdf_file, "status": "error", "error": str(e), "chunks": 0}
        else:
            logger.warning(f"No text extracted from: {pdf_file}")
            return {"file": pdf_file, "status": "no_text", "chunks": 0}
            
    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {e}")
        return {"file": pdf_file, "status": "error", "error": str(e), "chunks": 0}
    finally:
        logger.info(f"Completed processing: {pdf_file}")

async def process_single_document_with_semaphore(rag, pdf_file, semaphore):
    """Process a single PDF document with concurrency control."""
    async with semaphore:  # Limit concurrent operations
        return await process_single_document(rag, pdf_file)

async def process_all_documents_parallel(rag, pdf_files):
    # Create tasks for all documents
    tasks = [process_single_document(rag, pdf_file) for pdf_file in pdf_files]
    
    # Execute all tasks in parallel
    logger.info(f"Starting parallel processing of {len(tasks)} documents...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

async def process_all_documents_controlled_parallel(rag, pdf_files, max_concurrent=3):
    """Process documents with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all documents with semaphore
    tasks = [process_single_document_with_semaphore(rag, pdf_file, semaphore) for pdf_file in pdf_files]
    
    # Execute tasks with controlled concurrency
    logger.info(f"Starting controlled parallel processing of {len(tasks)} documents (max {max_concurrent} concurrent)...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

def cleanup_kv_store_encoding():
    """Clean up the kv store file to fix encoding issues."""
    kv_store_path = Path(WORKING_DIR) / "kv_store_doc_status.json"
    if kv_store_path.exists():
        try:
            # Try to read with different encodings
            doc_status = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(kv_store_path, 'r', encoding=encoding) as f:
                        doc_status = json.load(f)
                    logger.info(f"Successfully read kv store with {encoding} encoding")
                    break
                except Exception as e:
                    logger.warning(f"Failed to read kv store with {encoding}: {e}")
                    continue
            
            if doc_status:
                # Write back with proper UTF-8 encoding
                with open(kv_store_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_status, f, indent=2, ensure_ascii=False)
                logger.info("Fixed kv store encoding and saved with UTF-8")
            else:
                logger.error("Could not read kv store with any encoding, creating new one")
                # Create a new clean kv store
                with open(kv_store_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2)
        except Exception as e:
            logger.error(f"Error cleaning up kv store: {e}")

def main():
    logger.info("Starting RAG pipeline...")
    
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    logger.info("RAG instance initialized.")
    
    # Process all PDF files in the documents folder
    documents_dir = Path("Documents_Sample")
    if not documents_dir.exists():
        os.makedirs(documents_dir)
        logger.warning("Created documents directory. Please add PDF files to process.")
        return

    pdf_files = list(documents_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {documents_dir.resolve()}")
    
    # Load current processing status
    processing_status = load_processing_status()
    logger.info(f"Loaded processing status for {len(processing_status)} documents")
    
    # Filter out already processed documents
    unprocessed_files = []
    for pdf_file in pdf_files:
        if is_document_processed(pdf_file, processing_status):
            logger.info(f"Skipping already processed: {pdf_file}")
        else:
            unprocessed_files.append(pdf_file)
            logger.info(f"Will process: {pdf_file}")
    
    if not unprocessed_files:
        logger.info("All documents are already processed!")
        # Query the RAG system
        logger.info("Starting query...")
        response = rag.query(
            query="What is the main theme of the documents?",
            param=QueryParam(mode="mix", top_k=5, response_type="single line"),
        )
        logger.info("Query complete. Printing response.")
        print(response)
        return
    
    logger.info(f"Processing {len(unprocessed_files)} new documents out of {len(pdf_files)} total")
    
    # Process files in parallel using asyncio.gather
    # Choose your preferred method:
    # 1. Full parallel (all documents at once)
    # 2. Controlled parallel (limit concurrent operations)
    
    USE_CONTROLLED_PARALLEL = True  # Set to False for full parallel
    MAX_CONCURRENT_DOCUMENTS = 2    # Adjust based on your system/API limits
    
    if USE_CONTROLLED_PARALLEL:
        logger.info(f"Using controlled parallel processing (max {MAX_CONCURRENT_DOCUMENTS} concurrent)")
        results = asyncio.run(process_all_documents_controlled_parallel(rag, unprocessed_files, MAX_CONCURRENT_DOCUMENTS))
    else:
        logger.info("Using full parallel processing")
        results = asyncio.run(process_all_documents_parallel(rag, unprocessed_files))
    
    # Process results
    processed_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    skipped_count = len(results) - processed_count
    
    logger.info(f"Processing complete. {processed_count} files processed, {skipped_count} files skipped.")
    
    # Log detailed results
    for result in results:
        if isinstance(result, dict):
            logger.info(f"File: {result['file']} - Status: {result['status']}")
        else:
            logger.error(f"Exception in processing: {result}")
    
    # Query the RAG system
    logger.info("Starting query...")
    response = rag.query(
        query="Compare the performance of the Hospi BI and MarTech segments?",
        param=QueryParam(mode="mix", top_k=5, response_type="single line"),
    )
    logger.info("Query complete. Printing response.")
    print(response)

if __name__ == "__main__":
    main()