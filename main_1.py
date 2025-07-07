import os
import glob
import logging
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
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
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# File paths
ENHANCED_CHUNKS_FILE = "enhanced_temporal_chunks.json"
DOCUMENT_CHUNKS_DIR = "document_chunks"
TEMPORAL_INDEX_FILE = "temporal_index.json"
VECTOR_STORE_PATH = "enhanced_faiss_index"

class FinancialAnalysisState(TypedDict):
    """Enhanced state for financial analysis workflow"""
    pdf_files: List[str]
    current_file: Optional[str]
    current_page: Optional[int]
    extracted_pages: List[Dict]
    temporal_chunks: List['EnhancedTemporalChunk']
    query: Optional[str]
    query_intent: Optional[Dict]
    query_complexity: Optional[str]
    retrieved_chunks: List['EnhancedTemporalChunk']
    cfa_analysis: Optional[str]
    cfa_reasoning: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

@dataclass
class EnhancedTemporalChunk:
    """Enhanced chunk with comprehensive temporal and semantic metadata"""
    text: str
    month: str
    year: int
    page_number: int
    file_source: str
    document_name: str
    chunk_type: str  # 'table', 'narrative', 'header', 'summary', 'financial_metric', 'kpi', 'page_section'
    business_units: List[str]  # ['DaaS', 'Distribution', 'MarTech', 'Travel BI', 'Hospi BI', etc.]
    financial_metrics: List[str]  # ['revenue', 'ebitda', 'margin', 'growth', etc.]
    table_headers: List[str]  # Table column headers if applicable
    section_title: str  # Section heading (e.g., "Executive Summary", "P&L Analysis")
    contains_numbers: bool
    has_comparisons: bool  # YoY, budget vs actual, etc.
    period_references: List[str]  # ['Q1', 'Q2', 'H1', 'YTD', etc.]
    chunk_id: str
    confidence_score: float
    complexity_level: str  # 'simple', 'moderate', 'complex'
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class SimplifiedAnalysisState(TypedDict):
    """Simplified state for streamlined analysis workflow"""
    pdf_files: List[str]
    extracted_pages: List[Dict]
    temporal_chunks: List[EnhancedTemporalChunk]
    query: Optional[str]
    use_cfa: bool  # Simple toggle for CFA usage
    retrieved_chunks: List[EnhancedTemporalChunk]
    cfa_analysis: Optional[str]
    cfa_reasoning: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

class RateLimitManager:
    """Manage Gemini API rate limits"""
    def __init__(self, requests_per_minute=15):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0]) + 1
            if sleep_time > 0:
                logging.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        self.request_times.append(now)

class EnhancedFinancialRAG:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            client=genai, 
            temperature=0.1
        )
        self.cfa_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            client=genai,
            temperature=0.2
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07"
        )
        self.vector_store = None
        self.temporal_chunks: List[EnhancedTemporalChunk] = []
        self.rate_limiter = RateLimitManager()
        
        # Ensure document chunks directory exists
        os.makedirs(DOCUMENT_CHUNKS_DIR, exist_ok=True)

    def extract_temporal_info(self, filepath: str) -> Tuple[str, int]:
        """Extract month and year from filename using enhanced logic"""
        filename = os.path.basename(filepath).lower()
        
        # Enhanced month mapping including variations
        month_map = {
            'january': 'January', 'jan': 'January',
            'february': 'February', 'feb': 'February',
            'march': 'March', 'mar': 'March',
            'april': 'April', 'apr': 'April',
            'may': 'May',
            'june': 'June', 'jun': 'June',
            'july': 'July', 'jul': 'July',
            'august': 'August', 'aug': 'August',
            'september': 'September', 'sep': 'September', 'sept': 'September',
            'october': 'October', 'oct': 'October',
            'november': 'November', 'nov': 'November',
            'december': 'December', 'dec': 'December'
        }
        
        month = "Unknown"
        for key, value in month_map.items():
            if key in filename:
                month = value
                break
        
        # Try to extract year
        year_match = re.search(r'20\d{2}', filename)
        year = int(year_match.group()) if year_match else 2024
        
        return month, year

    def _process_page_with_semantic_chunking(self, page_data: Dict) -> List[EnhancedTemporalChunk]:
        """Process a single page with enhanced semantic understanding"""
        
        enhanced_chunking_prompt = f"""
        You are an expert financial analyst specializing in corporate financial reports. Analyze this page from a RateGain financial report and create intelligent semantic chunks.

        Document: {page_data['document_name']}
        Page: {page_data['page_number']}
        Month/Year: {page_data['month']} {page_data['year']}

        For each chunk, provide comprehensive analysis:

        1. **chunk_type**: Classify as one of:
           - 'table': Financial tables, data grids
           - 'executive_summary': High-level overview content
           - 'kpi_dashboard': Operational metrics, KPI summaries
           - 'financial_analysis': P&L, revenue analysis
           - 'trend_analysis': Growth, comparative data
           - 'section_header': Major section titles
           - 'narrative': Explanatory text

        2. **business_units**: Identify mentioned units:
           - DaaS (Travel BI, Hospi BI)
           - Distribution (DHISCO, RezGain, Enterprise Connectivity)
           - MarTech (Adara, BCV, MHS)

        3. **financial_metrics**: Extract specific metrics:
           - Revenue types (Gross, Net, GAAP)
           - Profitability (EBITDA, Margins)
           - Growth (YoY, QoQ)
           - Operational (New bookings, Churn, GRR, NRR)

        4. **table_headers**: If table, extract column headers
        5. **section_title**: Identify the section this belongs to
        6. **period_references**: Time periods mentioned (Q1, Q2, H1, YTD, etc.)
        7. **has_comparisons**: Boolean if contains comparative data
        8. **complexity_level**: 'simple' (basic metrics), 'moderate' (some analysis), 'complex' (deep financial analysis)

        **CRITICAL RULES:**
        - Keep complete tables together with their headers
        - Preserve numerical relationships and context
        - Don't split mid-sentence or mid-table
        - Group related financial sections
        - Maintain page-level context

        Return a JSON array with detailed metadata:

        Text to analyze:
        {page_data['text']}
        """
        
        try:
            response = self.llm.invoke(enhanced_chunking_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            if not isinstance(content, str):
                content = str(content)
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                chunk_data = json.loads(json_match.group())
                
                chunks = []
                for i, item in enumerate(chunk_data):
                    # Fallback: if 'text' is missing, try 'chunk_text', then fallback to page text
                    chunk_text = item.get("text") or item.get("chunk_text", "")
                    if not chunk_text:
                        logging.warning(f"Chunk {i} missing 'text'/'chunk_text' field, using page text as fallback.")
                        chunk_text = page_data['text']
                    
                    chunk = EnhancedTemporalChunk(
                        text=chunk_text,
                        month=page_data['month'],
                        year=page_data['year'],
                        page_number=page_data['page_number'],
                        file_source=page_data['file_source'],
                        document_name=page_data['document_name'],
                        chunk_type=item.get("chunk_type", "narrative"),
                        business_units=item.get("business_units", []),
                        financial_metrics=item.get("financial_metrics", []),
                        table_headers=item.get("table_headers", []),
                        section_title=item.get("section_title", ""),
                        contains_numbers=item.get("contains_numbers", False),
                        has_comparisons=item.get("has_comparisons", False),
                        period_references=item.get("period_references", []),
                        chunk_id=f"{page_data['document_name']}_p{page_data['page_number']}_c{i}",
                        confidence_score=item.get("confidence_score", 0.8),
                        complexity_level=item.get("complexity_level", "moderate")
                    )
                    chunks.append(chunk)
                
                return chunks
            else:
                # Fallback to page-based chunking
                return self._fallback_page_chunking(page_data)
                
        except Exception as e:
            logging.error(f"Error in semantic chunking for page {page_data['page_number']}: {e}")
            return self._fallback_page_chunking(page_data)

    def _fallback_page_chunking(self, page_data: Dict) -> List[EnhancedTemporalChunk]:
        """Fallback chunking based on page structure"""
        text = page_data['text']
        
        # Try to identify major sections
        section_patterns = [
            r'Section \d+:.*',
            r'[A-Z][a-z]+ Summary',
            r'Profit & Loss.*',
            r'Revenue.*',
            r'EBITDA.*',
            r'Key Performance.*'
        ]
        
        chunks = []
        current_chunk = ""
        current_section = "General"
        
        lines = text.split('\n')
        for line in lines:
            # Check if this line is a section header
            is_section = any(re.match(pattern, line.strip()) for pattern in section_patterns)
            
            if is_section and current_chunk:
                # Save current chunk
                chunk = self._create_fallback_chunk(current_chunk, page_data, current_section, len(chunks))
                chunks.append(chunk)
                current_chunk = line + '\n'
                current_section = line.strip()
            else:
                current_chunk += line + '\n'
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_fallback_chunk(current_chunk, page_data, current_section, len(chunks))
            chunks.append(chunk)
        
        return chunks

    def _create_fallback_chunk(self, text: str, page_data: Dict, section: str, chunk_index: int) -> EnhancedTemporalChunk:
        """Create a fallback chunk with basic metadata"""
        return EnhancedTemporalChunk(
            text=text,
            month=page_data['month'],
            year=page_data['year'],
            page_number=page_data['page_number'],
            file_source=page_data['file_source'],
            document_name=page_data['document_name'],
            chunk_type="narrative",
            business_units=self._extract_business_units(text),
            financial_metrics=self._extract_financial_metrics(text),
            table_headers=[],
            section_title=section,
            contains_numbers=bool(re.search(r'\d+', text)),
            has_comparisons=self._detect_comparisons(text),
            period_references=self._extract_period_references(text),
            chunk_id=f"{page_data['document_name']}_p{page_data['page_number']}_fb{chunk_index}",
            confidence_score=0.6,
            complexity_level="moderate"
        )

    def _detect_comparisons(self, text: str) -> bool:
        """Detect if text contains comparative analysis"""
        comparison_patterns = [
            r'vs\.?', r'compared to', r'year[- ]over[- ]year', r'y[- ]?o[- ]?y',
            r'growth', r'increase', r'decrease', r'higher', r'lower',
            r'budget.*actual', r'forecast.*actual'
        ]
        return any(re.search(pattern, text.lower()) for pattern in comparison_patterns)

    def _extract_period_references(self, text: str) -> List[str]:
        """Extract time period references"""
        periods = []
        period_patterns = {
            'Q1': r'\bq1\b|\bfirst quarter\b',
            'Q2': r'\bq2\b|\bsecond quarter\b',
            'Q3': r'\bq3\b|\bthird quarter\b',
            'Q4': r'\bq4\b|\bfourth quarter\b',
            'H1': r'\bh1\b|\bfirst half\b',
            'H2': r'\bh2\b|\bsecond half\b',
            'YTD': r'\bytd\b|\byear to date\b',
            'TTM': r'\bttm\b|\btrailing twelve months\b',
            'FY': r'\bfy\s*\d{2,4}\b|\bfiscal year\b'
        }
        
        text_lower = text.lower()
        for period, pattern in period_patterns.items():
            if re.search(pattern, text_lower):
                periods.append(period)
        
        return periods

    def _extract_business_units(self, text: str) -> List[str]:
        """Enhanced business unit extraction"""
        units = []
        unit_patterns = {
            'DaaS': r'\b(daas|data as a service)\b',
            'Travel BI': r'\btravel\s*bi\b',
            'Hospi BI': r'\bhospi\s*bi\b',
            'Distribution': r'\bdistribution\b',
            'DHISCO': r'\bdhisco\b',
            'RezGain': r'\brezgain\b',
            'Enterprise Connectivity': r'\benterprise\s*connectivity\b',
            'MarTech': r'\b(martech|marketing\s*technology)\b',
            'Adara': r'\badara\b',
            'BCV': r'\bbcv\b',
            'MHS': r'\bmhs\b'
        }
        
        text_lower = text.lower()
        for unit, pattern in unit_patterns.items():
            if re.search(pattern, text_lower):
                units.append(unit)
        
        return units

    def _extract_financial_metrics(self, text: str) -> List[str]:
        """Enhanced financial metrics extraction"""
        metrics = []
        metric_patterns = {
            'revenue': r'\brevenue\b',
            'gross_revenue': r'\bgross\s*revenue\b',
            'net_revenue': r'\bnet\s*revenue\b',
            'ebitda': r'\bebitda\b',
            'margin': r'\bmargin\b',
            'gross_margin': r'\bgross\s*margin\b',
            'ebitda_margin': r'\bebitda\s*margin\b',
            'cost': r'\bcost\b',
            'cogs': r'\bcogs\b|\bcost\s*of\s*goods\s*sold\b',
            'budget': r'\bbudget\b',
            'forecast': r'\bforecast\b',
            'actual': r'\bactual\b',
            'growth': r'\bgrowth\b',
            'yoy': r'\by[- ]?o[- ]?y\b|\byear\s*over\s*year\b',
            'profit': r'\bprofit\b',
            'earnings': r'\bearnings\b',
            'new_bookings': r'\bnew\s*bookings\b',
            'grr': r'\bgrr\b|\bgross\s*renewal\s*rate\b',
            'nrr': r'\bnrr\b|\bnet\s*renewal\s*rate\b',
            'churn': r'\bchurn\b',
            'ltv': r'\bltv\b|\blifetime\s*value\b',
            'cac': r'\bcac\b|\bcustomer\s*acquisition\s*cost\b'
        }
        
        text_lower = text.lower()
        for metric, pattern in metric_patterns.items():
            if re.search(pattern, text_lower):
                metrics.append(metric)
        
        return metrics

    def _save_document_chunks(self, document_name: str, new_chunks: List[EnhancedTemporalChunk]):
        """Save chunks for a specific document"""
        filename = os.path.join(DOCUMENT_CHUNKS_DIR, f"{document_name}_chunks.json")
        # Load existing chunks if file exists
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    existing_chunks = [EnhancedTemporalChunk.from_dict(item) for item in existing_data]
                except Exception:
                    existing_chunks = []
        else:
            existing_chunks = []
        
        # Append new chunks
        all_chunks = existing_chunks + new_chunks
        chunk_data = [chunk.to_dict() for chunk in all_chunks]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(new_chunks)} new chunks, {len(all_chunks)} total for document {document_name}")

    def build_knowledge_base(self):
        """Build the enhanced knowledge base incrementally"""
        documents_folder = os.path.join(os.getcwd(), "Documents_Sample")
        pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))
        
        # Check which documents already have chunks
        chunk_files = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(DOCUMENT_CHUNKS_DIR, "*_chunks.json")))
        new_pdf_files = []
        
        for pdf_path in pdf_files:
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            if f"{document_name}_chunks" not in chunk_files:
                new_pdf_files.append(pdf_path)

        # Only process new PDFs
        if new_pdf_files:
            logging.info(f"Found {len(new_pdf_files)} new PDF(s) to process")
            for pdf_path in new_pdf_files:
                document_name = os.path.splitext(os.path.basename(pdf_path))[0]
                try:
                    with open(pdf_path, "rb") as pdf:
                        pdf_reader = PdfReader(pdf)
                        
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                month, year = self.extract_temporal_info(pdf_path)
                                
                                page_data = {
                                    'text': page_text,
                                    'page_number': page_num,
                                    'file_source': pdf_path,
                                    'document_name': document_name,
                                    'month': month,
                                    'year': year,
                                    'total_pages': len(pdf_reader.pages)
                                }
                                
                                self.rate_limiter.wait_if_needed()
                                chunks = self._process_page_with_semantic_chunking(page_data)
                                self._save_document_chunks(document_name, chunks)
                                
                except Exception as e:
                    logging.error(f"Error processing {pdf_path}: {e}")
        else:
            logging.info("No new PDFs to process.")

        # Rebuild vector store from all chunk files
        all_chunks = []
        for chunk_file in glob.glob(os.path.join(DOCUMENT_CHUNKS_DIR, "*_chunks.json")):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                try:
                    chunk_data = json.load(f)
                    all_chunks.extend([EnhancedTemporalChunk.from_dict(item) for item in chunk_data])
                except Exception as e:
                    logging.error(f"Error loading chunks from {chunk_file}: {e}")
        
        self.temporal_chunks = all_chunks
        
        # Build vector store
        try:
            texts = []
            metadatas = []
            
            for chunk in all_chunks:
                # Create enhanced text representation
                enhanced_text = f"""
                Document: {chunk.document_name}
                Month: {chunk.month} {chunk.year}
                Page: {chunk.page_number}
                Section: {chunk.section_title}
                Business Units: {', '.join(chunk.business_units)}
                Financial Metrics: {', '.join(chunk.financial_metrics)}
                Period References: {', '.join(chunk.period_references)}
                Type: {chunk.chunk_type}
                
                Content:
                {chunk.text}
                """
                
                texts.append(enhanced_text)
                metadatas.append({
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "month": chunk.month,
                    "year": chunk.year,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type,
                    "business_units": chunk.business_units,
                    "financial_metrics": chunk.financial_metrics,
                    "section_title": chunk.section_title,
                    "complexity_level": chunk.complexity_level,
                    "has_comparisons": chunk.has_comparisons,
                    "period_references": chunk.period_references
                })
            
            # Create vector store
            self.vector_store = FAISS.from_texts(
                texts, 
                self.embeddings,
                metadatas=metadatas
            )
            
            # Save vector store and chunks
            self.vector_store.save_local(VECTOR_STORE_PATH)
            self._save_all_temporal_chunks(all_chunks)
            
            logging.info(f"Built enhanced vector store with {len(all_chunks)} chunks")
            return True
            
        except Exception as e:
            logging.error(f"Error building vector store: {e}")
            return False

    def _save_all_temporal_chunks(self, chunks: List[EnhancedTemporalChunk]):
        """Save all temporal chunks to master file"""
        with open(ENHANCED_CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump([chunk.to_dict() for chunk in chunks], f, indent=2, ensure_ascii=False)

    def _load_temporal_chunks(self) -> List[EnhancedTemporalChunk]:
        """Load temporal chunks from file"""
        if os.path.exists(ENHANCED_CHUNKS_FILE):
            with open(ENHANCED_CHUNKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [EnhancedTemporalChunk.from_dict(item) for item in data]
        return []

    def _find_chunk_by_id(self, chunk_id: str) -> Optional[EnhancedTemporalChunk]:
        """Find chunk by ID from loaded chunks"""
        for chunk in self.temporal_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def _simple_retrieval(self, query: str, k: int = 10) -> List[EnhancedTemporalChunk]:
        """Fast, simple retrieval for direct factual queries"""
        try:
            # Load vector store if not loaded
            if self.vector_store is None:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            # Simple similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Convert to TemporalChunks
            retrieved_chunks = []
            for doc, score in docs_with_scores:
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                original_chunk = self._find_chunk_by_id(chunk_id)
                if original_chunk:
                    retrieved_chunks.append(original_chunk)
            
            return retrieved_chunks
            
        except Exception as e:
            logging.error(f"Error in simple retrieval: {e}")
            return []

    def _enhanced_retrieval(self, query: str, k: int = 15) -> List[EnhancedTemporalChunk]:
        """Enhanced retrieval for CFA analysis"""
        try:
            # Load vector store if not loaded
            if self.vector_store is None:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            # Enhanced query construction for complex analysis
            enhanced_query = f"{query} analysis explanation reasoning factors impact trends comparison"
            
            # Retrieve more documents for comprehensive analysis
            docs_with_scores = self.vector_store.similarity_search_with_score(enhanced_query, k=k)
            
            # Enhanced filtering for CFA analysis
            filtered_docs = []
            for doc, score in docs_with_scores:
                metadata = doc.metadata
                relevance_boost = 0
                
                # Boost for complex content
                if metadata.get("complexity_level") == "complex":
                    relevance_boost += 0.3
                
                # Boost for comparative analysis
                if metadata.get("has_comparisons", False):
                    relevance_boost += 0.2
                
                # Boost for financial tables
                if metadata.get("chunk_type") == "table":
                    relevance_boost += 0.2
                
                adjusted_score = score - relevance_boost
                filtered_docs.append((doc, adjusted_score))
            
            # Sort by adjusted score
            filtered_docs.sort(key=lambda x: x[1])
            
            # Convert to TemporalChunks
            retrieved_chunks = []
            for doc, score in filtered_docs:
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                original_chunk = self._find_chunk_by_id(chunk_id)
                if original_chunk:
                    retrieved_chunks.append(original_chunk)
            
            return retrieved_chunks
            
        except Exception as e:
            logging.error(f"Error in enhanced retrieval: {e}")
            return []

    def _run_chartered_financial_analyst(self, query: str, chunks: List[EnhancedTemporalChunk], chat_history: Optional[List[dict]] = None) -> Tuple[Optional[str], Optional[str]]:
        """Chartered Financial Analyst for complex analysis, context aware."""
        if chat_history is None:
            chat_history = []
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"""
            --- {chunk.document_name} | {chunk.month} {chunk.year} | Page {chunk.page_number} | {chunk.section_title} ---
            Business Units: {', '.join(chunk.business_units)}
            Financial Metrics: {', '.join(chunk.financial_metrics)}
            Type: {chunk.chunk_type} | Complexity: {chunk.complexity_level}
            {chunk.text}
            """)
        context = "\n\n".join(context_parts)
        # Add chat history context
        history_str = "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
            for msg in chat_history if msg['role'] in ('user', 'assistant')
        ])
        cfa_prompt = f"""
        You are a senior Chartered Financial Analyst (CFA) with 15+ years of experience in financial analysis, 
        corporate finance, and strategic consulting. You specialize in SaaS company analysis and travel technology sector.
        **CRITICAL INSTRUCTION**: Do NOT make up or estimate any numbers. Only use numbers, values, or facts that are explicitly present in the provided context. If a number is not found, state that it is not available. Never attempt to calculate or guess values. If the user asks for a calculation, only perform it if all required numbers are present in the context.
        Use <thinking> tags to show your analytical reasoning process step by step. This thinking should be visible to demonstrate your analytical methodology.
        
        **Chat History:**
        {history_str}
        
        Query: {query}
        <thinking>
        Let me analyze this query systematically:
        1. What is the core financial question being asked?
        2. What data do I need to examine?
        3. What analytical frameworks should I apply?
        4. What are the potential root causes or factors?
        5. What are the business implications?
        6. What recommendations can I provide?
        </thinking>
        Provide a comprehensive CFA-level analysis that includes:
        **Financial Analysis Framework:**
        1. **Data Assessment**: What the numbers tell us
        2. **Trend Analysis**: Patterns and trajectories  
        3. **Comparative Analysis**: Benchmarking and context
        4. **Root Cause Analysis**: Why these results occurred
        5. **Risk Assessment**: Potential concerns and red flags
        6. **Strategic Implications**: Business impact and meaning
        7. **Recommendations**: Actionable next steps
        **Context:**
        {context}
        **Your CFA Analysis:**
        """
        try:
            self.rate_limiter.wait_if_needed()
            response = self.cfa_llm.invoke(cfa_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            if not isinstance(content, str):
                content = str(content)
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            thinking = thinking_match.group(1).strip() if thinking_match else ""
            analysis = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL).strip()
            return thinking, analysis
        except Exception as e:
            logging.error(f"Error in CFA analysis: {e}")
            return None, None

    def _generate_simple_response(self, query: str, chunks: List[EnhancedTemporalChunk], chat_history: Optional[List[dict]] = None) -> str:
        """Generate simple, direct response for factual queries. Uses chat_history for context."""
        if chat_history is None:
            chat_history = []
        # Prepare context
        context_parts = []
        for chunk in chunks[:8]:  # Top 8 chunks for simple queries
            context_parts.append(f"""
            --- {chunk.month} {chunk.year} | {chunk.section_title} ---
            {chunk.text}
            """)
        context = "\n\n".join(context_parts)
        # Add chat history context
        history_str = "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
            for msg in chat_history if msg['role'] in ('user', 'assistant')
        ])
        simple_prompt = f"""
        You are an expert financial analyst with detailed knowledge of RateGain Travel Technologies.
        
        **CRITICAL INSTRUCTION**: Do NOT make up or estimate any numbers. Only use numbers, values, or facts that are explicitly present in the provided Financial Data Context. If a number is not found, state that it is not available. Never attempt to calculate or guess values. If the user asks for a calculation, only perform it if all required numbers are present in the context.
        
        **User Query:** {query}
        
        **Chat History:**
        {history_str}
        
        **Financial Data Context:**
        {context}
        
        Provide a clear, direct answer that includes:
        1. **Direct Answer**: Address the specific question clearly
        2. **Key Financial Metrics**: Present relevant numbers (only if present in context)
        3. **Context**: Brief background if helpful
        4. **Source Reference**: Mention the time period/document if relevant
        
        Keep the response concise and factual. Use specific numbers where available. If any information is missing, clearly state what is not available.
        
        **Analysis:**
        """
        try:
            self.rate_limiter.wait_if_needed()
            response = self.llm.invoke(simple_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            if not isinstance(content, str):
                content = str(content)
            return content
        except Exception as e:
            logging.error(f"Error generating simple response: {e}")
            return f"Error processing query: {str(e)}"

    def _generate_complex_response(self, query: str, cfa_analysis: str, chat_history: Optional[List[dict]] = None) -> str:
        """Generate complex response incorporating CFA analysis and chat history."""
        if chat_history is None:
            chat_history = []
        # Add chat history context
        history_str = "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
            for msg in chat_history if msg['role'] in ('user', 'assistant')
        ])
        complex_prompt = f"""
        You are a senior financial analyst providing insights based on comprehensive CFA analysis.
        
        **CRITICAL INSTRUCTION**: Do NOT make up or estimate any numbers. Only use numbers, values, or facts that are explicitly present in the CFA Expert Analysis or the provided context. If a number is not found, state that it is not available. Never attempt to calculate or guess values. If the user asks for a calculation, only perform it if all required numbers are present in the context.
        
        **Chat History:**
        {history_str}
        
        A Chartered Financial Analyst has provided the following expert analysis:
        
        **CFA Expert Analysis:**
        {cfa_analysis}
        
        **User Query:** {query}
        
        Synthesize the CFA analysis to provide a comprehensive response that:
        1. **Executive Summary**: Key findings and direct answer
        2. **Financial Analysis**: Detailed insights from the data
        3. **Root Cause Analysis**: Why these results occurred
        4. **Business Implications**: What this means for the company
        5. **Strategic Recommendations**: Actionable next steps
        
        Structure your response clearly with headers and bullet points for readability.
        Include specific numbers and percentages where available (only if present in context).
        
        **Comprehensive Analysis:**
        """
        try:
            self.rate_limiter.wait_if_needed()
            response = self.llm.invoke(complex_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            if not isinstance(content, str):
                content = str(content)
            return content
        except Exception as e:
            logging.error(f"Error generating complex response: {e}")
            return f"Error processing query: {str(e)}"

    def process_query(self, user_question: str, use_cfa: bool, chat_history: Optional[List[dict]] = None) -> dict:
        """Process a user query using either simple or CFA (enhanced) flow based on toggle. Accepts chat_history for context awareness."""
        if not self.temporal_chunks:
            self.temporal_chunks = self._load_temporal_chunks()
        if chat_history is None:
            chat_history = []
        log = logging.info
        log(f"[Query] use_cfa={use_cfa}")
        result = {
            "response": None,
            "cfa_reasoning": None,
            "cfa_analysis": None,
            "use_cfa": use_cfa,
            "chunks_used": 0,
            "error": None
        }
        query_path = ["Retrieval"]
        try:
            if use_cfa:
                retrieved_chunks = self._enhanced_retrieval(user_question, k=15)
                result["chunks_used"] = len(retrieved_chunks)
                log("[Node] CFA Analysis")
                query_path.append("CFA Analysis")
                thinking, analysis = self._run_chartered_financial_analyst(user_question, retrieved_chunks, chat_history)
                result["cfa_reasoning"] = thinking
                result["cfa_analysis"] = analysis
                log("[Node] Complex Response Generation")
                query_path.append("Complex Response Generation")
                response = self._generate_complex_response(user_question, analysis or "", chat_history)
                result["response"] = response
            else:
                retrieved_chunks = self._simple_retrieval(user_question, k=10)
                result["chunks_used"] = len(retrieved_chunks)
                log("[Node] Simple Response Generation")
                query_path.append("Simple Response Generation")
                response = self._generate_simple_response(user_question, retrieved_chunks, chat_history)
                result["response"] = response
        except Exception as e:
            log(f"Error processing query: {e}")
            result["error"] = str(e)
        log(f"[Query Path] {' -> '.join(query_path)}")
        return result

# Simplified Streamlit Application
def main():
    st.set_page_config(
        page_title="Enhanced Financial RAG with CFA Toggle", 
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏢 Enhanced Financial Reports Analysis")
    st.markdown("*Streamlined RAG with Optional CFA Intelligence Toggle*")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedFinancialRAG()
    
    rag_system = st.session_state.rag_system
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Build knowledge base button
        if st.button("🔨 Build Enhanced Knowledge Base", type="primary"):
            with st.spinner("Building enhanced knowledge base..."):
                success = rag_system.build_knowledge_base()
                if success:
                    st.success("✅ Enhanced knowledge base built successfully!")
                else:
                    st.error("❌ Error building knowledge base")
        
        # Show enhanced system stats
        if rag_system.temporal_chunks:
            st.metric("Total Enhanced Chunks", len(rag_system.temporal_chunks))
            
            # Document distribution
            doc_counts = {}
            complexity_counts = {}
            chunk_type_counts = {}
            
            for chunk in rag_system.temporal_chunks:
                doc_counts[chunk.document_name] = doc_counts.get(chunk.document_name, 0) + 1
                complexity_counts[chunk.complexity_level] = complexity_counts.get(chunk.complexity_level, 0) + 1
                chunk_type_counts[chunk.chunk_type] = chunk_type_counts.get(chunk.chunk_type, 0) + 1
            
            st.subheader("📊 Analytics Dashboard")
            
            # Document distribution
            st.write("**📄 Document Distribution:**")
            for doc, count in sorted(doc_counts.items()):
                st.text(f"{doc}: {count} chunks")
            
            # Complexity distribution
            st.write("**🧠 Complexity Distribution:**")
            for complexity, count in complexity_counts.items():
                st.text(f"{complexity.title()}: {count} chunks")
            
            # Chunk type distribution
            st.write("**📑 Content Type Distribution:**")
            for chunk_type, count in chunk_type_counts.items():
                st.text(f"{chunk_type.replace('_', ' ').title()}: {count} chunks")
        
        st.divider()
        
        # Processing Mode Information
        st.subheader("🔧 Processing Modes")
        st.info("**Fast Mode**: Direct factual answers")
        st.info("**CFA Mode**: Deep analytical insights with reasoning")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "I'm ready to analyze your financial data! Use the toggle to choose between fast factual answers or deep CFA analysis."}
            ]
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm ready to analyze your financial data! Use the toggle to choose between fast factual answers or deep CFA analysis."}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "cfa_reasoning" in message:
                # Display processing mode indicator
                if message.get("use_cfa"):
                    st.caption("🎓 CFA Analysis Mode")
                    # Display CFA reasoning if available
                    if message.get("cfa_reasoning"):
                        with st.expander("🧠 CFA Analytical Reasoning", expanded=False):
                            st.write(message["cfa_reasoning"])
                else:
                    st.caption("⚡ Fast Analysis Mode")
                st.write(message["content"])
                # Show performance info
                if message.get("chunks_used"):
                    st.caption(f"📄 Analyzed {message['chunks_used']} document chunks")
            else:
                st.write(message["content"])

    # --- Move chat input and toggle to the very end of the main function ---
    col1, col2 = st.columns([6, 1])
    with col1:
        prompt = st.chat_input("Ask about financial metrics, trends, or analysis...")
    with col2:
        use_cfa = st.toggle("🎓 CFA", value=False, help="Enable for deep analytical insights with reasoning")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        # Generate response, pass chat history (excluding system messages)
        chat_history = [msg for msg in st.session_state.messages if msg["role"] in ("user", "assistant")]
        with st.chat_message("assistant"):
            # Show processing mode
            if use_cfa:
                st.caption("🎓 CFA Analysis Mode")
                with st.spinner("Analyzing with CFA intelligence..."):
                    result = rag_system.process_query(prompt, use_cfa=True, chat_history=chat_history)
            else:
                st.caption("⚡ Fast Analysis Mode")
                with st.spinner("Retrieving financial data..."):
                    result = rag_system.process_query(prompt, use_cfa=False, chat_history=chat_history)
            # Display CFA reasoning if available
            if result.get("cfa_reasoning"):
                with st.expander("🧠 CFA Analytical Reasoning", expanded=True):
                    st.write(result["cfa_reasoning"])
            # Display main response
            response = result["response"]
            st.write(response)
            # Show performance info
            if result.get("chunks_used"):
                st.caption(f"📄 Analyzed {result['chunks_used']} document chunks")
            # Handle errors
            if result.get("error"):
                st.error(f"Error: {result['error']}")
        # Add enhanced assistant message
        assistant_message = {
            "role": "assistant", 
            "content": result["response"],
            "cfa_reasoning": result.get("cfa_reasoning"),
            "use_cfa": use_cfa,
            "chunks_used": result.get("chunks_used", 0)
        }
        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()