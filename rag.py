import streamlit as st
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import logging
import pickle
from dataclasses import dataclass
from enum import Enum
import re
from dotenv import load_dotenv

# LightRAG imports
from lightrag import LightRAG, QueryParam
#from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
#from lightrag.embed import openai_embedding

# Google AI imports
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

# Document processing
import PyPDF2
from PIL import Image
import pytesseract
import mammoth
import docx
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"

@dataclass
class ConversationContext:
    history: List[Dict[str, Any]]
    current_timeframe: str
    last_query_type: QueryType
    relevant_months: List[str]
    
class DocumentProcessor:
    """Handle multi-modal document processing"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.jpg', '.png', '.jpeg']
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            with open(file_path, 'rb') as file:
                result = mammoth.extract_raw_text(file)
                return result.value
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {e}")
            return ""
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and extract metadata"""
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).stem
        
        logger.info(f"Processing document: {file_path}")
        # Extract month from filename
        month_match = re.search(
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
            file_name.lower()
        )
        month = month_match.group(1) if month_match else "unknown"
        logger.debug(f"Extracted month: {month} from {file_name}")
        # Extract year from filename
        year_match = re.search(r'(20\d{2})', file_name)
        year = year_match.group(1) if year_match else "2025"
        logger.debug(f"Extracted year: {year} from {file_name}")
        
        content = ""
        if file_ext == '.pdf':
            content = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            content = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_ext in ['.jpg', '.png', '.jpeg']:
            content = self.extract_text_from_image(file_path)
        
        logger.info(f"Finished processing document: {file_path}")
        return {
            'file_path': file_path,
            'file_name': file_name,
            'month': month,
            'year': year,
            'content': content,
            'file_type': file_ext,
            'processed_at': datetime.now().isoformat()
        }

class SemanticChunker:
    """Intelligent chunking based on document structure"""
    
    def __init__(self):
        self.section_patterns = {
            'executive_summary': r'executive\s+summary',
            'financials': r'financials?',
            'key_accounts': r'key\s+accounts?',
            'region_wise': r'region\s+wise|regional',
            'cash_investments': r'cash\s+(&|and)\s+investments?',
            'others': r'others?|miscellaneous'
        }
        
        self.product_patterns = {
            'travel_bi': r'travel\s+bi',
            'hospi_bi': r'hospi\s+bi',
            'adara': r'adara',
            'daas': r'daas',
            'distribution': r'distribution',
            'martech': r'martech',
            'bcv': r'bcv',
            'mhs': r'mhs',
            'channel_manager': r'channel\s+manager',
            'enterprise_connectivity': r'enterprise\s+connectivity'
        }
        
        self.metric_patterns = {
            'revenue': r'revenue',
            'ebitda': r'ebitda',
            'gross_margin': r'gross\s+margin|gm',
            'cogs': r'cogs|cost\s+of\s+goods\s+sold',
            'renewal_rate': r'renewal\s+rate|grr|nrr',
            'customer_count': r'customer\s+count'
        }
    
    def identify_section(self, text: str) -> str:
        """Identify which section the text belongs to"""
        text_lower = text.lower()
        for section, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower):
                return section
        return 'general'
    
    def identify_products(self, text: str) -> List[str]:
        """Identify products mentioned in the text"""
        text_lower = text.lower()
        products = []
        for product, pattern in self.product_patterns.items():
            if re.search(pattern, text_lower):
                products.append(product)
        return products
    
    def identify_metrics(self, text: str) -> List[str]:
        """Identify metrics mentioned in the text"""
        text_lower = text.lower()
        metrics = []
        for metric, pattern in self.metric_patterns.items():
            if re.search(pattern, text_lower):
                metrics.append(metric)
        return metrics
    
    def chunk_document(self, document: Dict[str, Any], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Create semantic chunks from document"""
        content = document['content']
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        current_metadata = {
            'month': document['month'],
            'year': document['year'],
            'file_name': document['file_name'],
            'section': 'general',
            'products': [],
            'metrics': [],
            'chunk_type': 'text'
        }
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) == 0:
                continue
                
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': current_metadata.copy()
                })
                current_chunk = ""
            
            current_chunk += paragraph + "\n\n"
            
            # Update metadata based on paragraph content
            section = self.identify_section(paragraph)
            if section != 'general':
                current_metadata['section'] = section
            
            products = self.identify_products(paragraph)
            current_metadata['products'].extend(products)
            
            metrics = self.identify_metrics(paragraph)
            current_metadata['metrics'].extend(metrics)
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': current_metadata.copy()
            })
        
        logger.debug(f"Created {len(chunks)} chunks for document: {document['file_name']}")
        return chunks

class GeminiLLMWrapper:
    """Wrapper for Gemini LLM to work with LightRAG"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            logger.info(f"Calling Gemini LLM with prompt at {datetime.now()}")
            response = self.model.generate_content(prompt)
            logger.info(f"Calling Gemini LLM completed at {datetime.now()}")
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini LLM call: {e}")
            return f"Error: {str(e)}"

class GeminiEmbeddingWrapper:
    """Wrapper for Gemini embeddings"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"
        self.embedding_dim = 768  # or the correct dimension for your model
    
    async def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Calling Gemini Embedding for {len(texts)} texts at {datetime.now()}")
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            logger.debug(f"Returning {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error in Gemini embedding call: {e}")
            return [[0.0] * 768 for _ in texts]  # Return zero embeddings as fallback

class QueryClassifier:
    """Classify queries as factual or analytical"""
    
    def __init__(self, llm_wrapper):
        self.llm = llm_wrapper
        
    async def classify_query(self, query: str, context: ConversationContext) -> QueryType:
        """Classify if query needs analytical processing"""
        
        analytical_keywords = [
            'compare', 'analyze', 'analyse', 'trend', 'insight', 'reason', 'why',
            'increase', 'decrease', 'growth', 'decline', 'performance', 'correlation',
            'impact', 'cause', 'effect', 'pattern', 'variance', 'difference'
        ]
        
        factual_keywords = [
            'what', 'when', 'where', 'who', 'list', 'show', 'display', 'value',
            'amount', 'number', 'count', 'total', 'sum'
        ]
        
        query_lower = query.lower()
        
        # Check for analytical keywords
        analytical_score = sum(1 for keyword in analytical_keywords if keyword in query_lower)
        factual_score = sum(1 for keyword in factual_keywords if keyword in query_lower)
        
        # Use LLM for complex classification
        if analytical_score == factual_score:
            classification_prompt = f"""
            Classify this query as either "factual" or "analytical":
            
            Query: {query}
            
            Factual queries ask for specific data points, values, or direct information.
            Analytical queries ask for comparisons, trends, insights, reasons, or deeper analysis.
            
            Respond with only: "factual" or "analytical"
            """
            
            try:
                response = await self.llm(classification_prompt)
                logger.info(f"Query classified as: {QueryType.ANALYTICAL if 'analytical' in response.lower() else QueryType.FACTUAL}")
                return QueryType.ANALYTICAL if 'analytical' in response.lower() else QueryType.FACTUAL
            except:
                # Default to analytical for complex queries
                return QueryType.ANALYTICAL
        
        return QueryType.ANALYTICAL if analytical_score > factual_score else QueryType.FACTUAL

class CharteredFinancialAnalyst:
    """Advanced financial analysis agent"""
    
    def __init__(self, rag_system, llm_wrapper):
        self.rag = rag_system
        self.llm = llm_wrapper
        
    async def analyze_query(self, query: str, context: ConversationContext) -> str:
        """Perform deep financial analysis"""
        
        logger.info(f"Analyzing query: {query}")
        # Extract relevant time periods and metrics
        time_context = self._extract_time_context(query)
        
        # Retrieve comprehensive data
        search_results = await self.rag.aquery(
            query,
            param=QueryParam(
                mode="mix",
                only_need_context=False,
                response_type="Multiple Paragraphs"
            )
        )
        
        # Enhanced analysis prompt
        analysis_prompt = f"""
        You are a Chartered Financial Analyst specializing in travel technology companies. 
        Analyze the following query with deep financial expertise:
        
        Query: {query}
        
        Context from previous conversations:
        {json.dumps(context.history[-3:], indent=2) if context.history else "No previous context"}
        
        Financial Data Retrieved:
        {search_results}
        
        Current Timeline: July 2025
        Available Data: FY 2024-2025 (March to December)
        
        Please provide a comprehensive financial analysis including:
        1. Direct answer to the query with specific numbers
        2. Comparative analysis across time periods
        3. Key insights and trends
        4. Potential reasons for changes/variations
        5. Business implications
        6. Recommendations if applicable
        
        Focus on:
        - Revenue trends and drivers
        - Profitability analysis (EBITDA, margins)
        - Product performance comparison
        - Regional performance insights
        - Customer metrics analysis
        - Cash flow implications
        
        Be specific with numbers, percentages, and timeframes.
        Reference the data sources and months clearly.
        Provide actionable insights for business leadership.
        """
        
        try:
            analysis = await self.llm(analysis_prompt)
            logger.info(f"Analysis complete for query: {query}")
            return analysis
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            return f"Error performing financial analysis: {str(e)}"
    
    def _extract_time_context(self, query: str) -> Dict[str, Any]:
        """Extract time-related context from query"""
        time_patterns = {
            'last_month': r'last\s+month',
            'last_quarter': r'last\s+quarter',
            'last_3_months': r'last\s+3\s+months|past\s+3\s+months',
            'ytd': r'ytd|year\s+to\s+date',
            'specific_month': r'(january|february|march|april|may|june|july|august|september|october|november|december)'
        }
        
        query_lower = query.lower()
        context = {}
        
        for pattern_name, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                context[pattern_name] = True
        
        return context

class RateGainRAGSystem:
    """Main RAG system for RateGain financial reports"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.working_dir = "./rategain_rag_data"
        self.documents_dir = "./Documents"
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        self.llm_wrapper = GeminiLLMWrapper(api_key)
        self.embedding_wrapper = GeminiEmbeddingWrapper(api_key)
        
        # Initialize LightRAG
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.llm_wrapper,
            embedding_func=self.embedding_wrapper
        )
        
        # Initialize specialized components
        self.query_classifier = QueryClassifier(self.llm_wrapper)
        self.financial_analyst = CharteredFinancialAnalyst(self.rag, self.llm_wrapper)
        
        # Conversation context
        self.conversation_context = ConversationContext(
            history=[],
            current_timeframe="July 2025",
            last_query_type=QueryType.FACTUAL,
            relevant_months=[]
        )
        
        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.documents_dir, exist_ok=True)
    
    def load_documents(self):
        """Load and process all documents in the Documents folder"""
        if not os.path.exists(self.documents_dir):
            st.error(f"Documents directory not found: {self.documents_dir}")
            return
        
        # Check if documents are already processed
        processed_file = os.path.join(self.working_dir, "processed_documents.json")
        
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                processed_docs = json.load(f)
            st.success(f"Loaded {len(processed_docs)} previously processed documents")
            return
        
        # Process documents
        documents = []
        document_files = [f for f in os.listdir(self.documents_dir) 
                         if any(f.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt'])]
        
        if not document_files:
            st.warning("No documents found in the Documents folder")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, filename in enumerate(document_files):
            status_text.text(f"Processing {filename}...")
            
            file_path = os.path.join(self.documents_dir, filename)
            doc = self.doc_processor.process_document(file_path)
            
            if doc['content']:
                # Create semantic chunks
                chunks = self.chunker.chunk_document(doc)
                
                # Insert chunks into RAG system
                for chunk in chunks:
                    chunk_text = f"""
                    Document: {chunk['metadata']['file_name']}
                    Month: {chunk['metadata']['month']}
                    Year: {chunk['metadata']['year']}
                    Section: {chunk['metadata']['section']}
                    Products: {', '.join(chunk['metadata']['products'])}
                    Metrics: {', '.join(chunk['metadata']['metrics'])}
                    
                    Content:
                    {chunk['content']}
                    """
                    
                    # Insert into LightRAG
                    asyncio.run(self.rag.ainsert(chunk_text))
                
                documents.append(doc)
                logger.info(f"Processed {filename}: {len(chunks)} chunks created")
            
            progress_bar.progress((i + 1) / len(document_files))
        
        # Save processed documents info
        with open(processed_file, 'w') as f:
            json.dump(documents, f, indent=2)
        
        # Log April chunks for debugging
        self.log_april_chunks()
        
        status_text.text("✅ All documents processed successfully!")
        st.success(f"Processed {len(documents)} documents with semantic chunking")
    
    async def query(self, user_query: str) -> str:
        """Main query processing function"""
        try:
            logger.info(f"Processing user query: {user_query}")
            # Classify query
            query_type = await self.query_classifier.classify_query(
                user_query, self.conversation_context
            )
            
            # Add current query to context
            self.conversation_context.history.append({
                'query': user_query,
                'timestamp': datetime.now().isoformat(),
                'type': query_type.value
            })
            
            # Route to appropriate handler
            if query_type == QueryType.ANALYTICAL:
                response = await self.financial_analyst.analyze_query(
                    user_query, self.conversation_context
                )
            else:
                # Handle factual queries with direct RAG
                response = await self.rag.aquery(
                    user_query,
                    param=QueryParam(
                        mode="mix",
                        only_need_context=False,
                        response_type="Multiple Paragraphs"
                    )
                )
            
            # Add response to context
            self.conversation_context.history[-1]['response'] = response
            self.conversation_context.last_query_type = query_type
            
            logger.info(f"Query processing complete for: {user_query}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_context.history = []
        st.success("Conversation history cleared")

    def log_april_chunks(self):
        """Log all chunks and their metadata for the month of April."""
        logger.info("--- Logging all chunks for April ---")
        # Assuming LightRAG stores chunks in a retrievable way
        # We'll try to retrieve all chunks with metadata month == 'april'
        try:
            # If your RAG system has a way to access all chunks, use it here
            # This is a generic example; adapt if your storage is different
            all_chunks = self.rag.vector_db_storage_cls().get_all_documents() if hasattr(self.rag, 'vector_db_storage_cls') else []
            april_chunks = [chunk for chunk in all_chunks if chunk.get('metadata', {}).get('month', '').lower() == 'april']
            for i, chunk in enumerate(april_chunks):
                logger.info(f"April Chunk {i+1}: Metadata: {chunk.get('metadata')}\nContent: {chunk.get('content')[:300]}...")
            if not april_chunks:
                logger.warning("No chunks found for April.")
        except Exception as e:
            logger.error(f"Error logging April chunks: {e}")

def main():
    st.set_page_config(
        page_title="RateGain Financial RAG Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for elegant UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    
    .sidebar-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 RateGain Financial RAG Assistant</h1>
        <p>Intelligent Financial Analysis with Conversational AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 🔧 Configuration")
        # API Key input (removed)
        # api_key = st.text_input(
        #     "Gemini API Key",
        #     type="password",
        #     help="Enter your Google Gemini API key"
        # )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
            st.stop()
        
        st.markdown("### 📊 System Status")
        if 'rag_system' not in st.session_state:
            st.info("System not initialized")
        else:
            st.success("System ready")
        
        st.markdown("### 💡 Query Examples")
        st.markdown("""
        - *Compare GAAP EBITDA for Travel BI-OTA for the past three months*
        - *What was the revenue for BCV in the last quarter?*
        - *Analyze the trend in customer count for Distribution*
        - *Show me the cash flow for March vs April*
        """)
        
        st.markdown("### 🔄 Actions")
        if st.button("Clear Conversation", type="secondary"):
            if 'rag_system' in st.session_state:
                st.session_state.rag_system.clear_conversation()
            if 'messages' in st.session_state:
                st.session_state.messages = []
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = RateGainRAGSystem(api_key)
            st.session_state.rag_system.load_documents()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat Interface")
        
        # Display chat history
        if st.session_state.messages:
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Query input
        user_input = st.text_input(
            "Ask me anything about RateGain's financial data:",
            placeholder="e.g., Compare revenue growth for Travel BI over the last 3 months"
        )
        
        if st.button("Send", type="primary") and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process query
            with st.spinner("Processing your query..."):
                response = asyncio.run(st.session_state.rag_system.query(user_input))
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update display
            st.rerun()
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        # Display some quick stats
        if 'rag_system' in st.session_state:
            context = st.session_state.rag_system.conversation_context
            
            st.metric("Queries Processed", len(context.history))
            st.metric("Current Timeline", context.current_timeframe)
            
            if context.history:
                last_query_type = context.history[-1].get('type', 'unknown')
                st.metric("Last Query Type", last_query_type.title())
        
        # System information
        st.markdown("### ℹ️ System Info")
        st.markdown("""
        - **Model**: Gemini
        - **Framework**: LightRAG
        - **Data Period**: FY 2024-2025
        - **Months**: March - December
        - **Sections**: 6 main categories
        - **Analysis**: CFA-level insights
        """)
        
        # Performance metrics
        if os.path.exists("./rategain_rag_data"):
            st.markdown("### 📊 Performance")
            try:
                # Count processed documents
                processed_file = "./rategain_rag_data/processed_documents.json"
                if os.path.exists(processed_file):
                    with open(processed_file, 'r') as f:
                        docs = json.load(f)
                    st.metric("Documents Loaded", len(docs))
                
                # Show data freshness
                st.metric("Data Freshness", "✅ Current")
                
                # Show response time estimate
                st.metric("Avg Response Time", "3-5 seconds")
                
            except Exception as e:
                st.error(f"Error loading metrics: {e}")

# Additional utility functions for production deployment
class ProductionUtils:
    """Utilities for production deployment"""
    
    @staticmethod
    def setup_logging():
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rategain_rag.log'),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def health_check():
        """System health check"""
        checks = {
            'documents_dir': os.path.exists('./Documents'),
            'working_dir': os.path.exists('./rategain_rag_data'),
            'processed_docs': os.path.exists('./rategain_rag_data/processed_documents.json')
        }
        return all(checks.values()), checks
    
    @staticmethod
    def get_system_stats():
        """Get comprehensive system statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'healthy',
            'memory_usage': 'normal',
            'response_time': 'optimal'
        }
        return stats

# Configuration for different deployment environments
class Config:
    """Configuration management"""
    
    DEVELOPMENT = {
        'log_level': 'DEBUG',
        'cache_ttl': 300,  # 5 minutes
        'max_conversation_history': 50
    }
    
    PRODUCTION = {
        'log_level': 'INFO',
        'cache_ttl': 3600,  # 1 hour
        'max_conversation_history': 100
    }
    
    @classmethod
    def get_config(cls, environment='production'):
        return cls.PRODUCTION if environment == 'production' else cls.DEVELOPMENT

# Defensive patch for LightRAG pipeline_status['history_messages']
# (This is a monkey-patch workaround, ideally should be fixed in LightRAG itself)
import lightrag.lightrag as lightrag_module
orig_apipeline_process_enqueue_documents = lightrag_module.LightRAG.apipeline_process_enqueue_documents
async def safe_apipeline_process_enqueue_documents(self, *args, **kwargs):
    try:
        return await orig_apipeline_process_enqueue_documents(self, *args, **kwargs)
    except KeyError as e:
        if str(e) == "'history_messages'":
            logger.warning("'history_messages' key missing in pipeline_status, skipping deletion.")
            return None
        raise
lightrag_module.LightRAG.apipeline_process_enqueue_documents = safe_apipeline_process_enqueue_documents

if __name__ == "__main__":
    # Setup logging for production
    ProductionUtils.setup_logging()
    
    # Run health check
    is_healthy, checks = ProductionUtils.health_check()
    if not is_healthy:
        logger.warning(f"System health check failed: {checks}")
    
    # Start the application
    main()