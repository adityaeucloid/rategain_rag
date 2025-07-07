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

class FinancialAnalysisState(TypedDict):
    """Enhanced state for financial analysis workflow"""
    pdf_files: List[str]
    current_file: Optional[str]
    current_page: Optional[int]
    extracted_pages: List[Dict]
    temporal_chunks: List[EnhancedTemporalChunk]
    query: Optional[str]
    query_intent: Optional[Dict]
    query_complexity: Optional[str]
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
            temperature=0.2  # Slightly higher for analytical creativity
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07"
        )
        self.vector_store = None
        self.temporal_chunks: List[EnhancedTemporalChunk] = []
        self.temporal_index = {}
        self.workflow = self._create_workflow()
        self.rate_limiter = RateLimitManager()
        
        # Ensure document chunks directory exists
        os.makedirs(DOCUMENT_CHUNKS_DIR, exist_ok=True)
        
    def _create_workflow(self):
        """Create enhanced LangGraph workflow"""
        workflow = StateGraph(FinancialAnalysisState)
        
        # Add nodes for document processing
        workflow.add_node("extract_files", self._extract_pdf_files)
        workflow.add_node("extract_pages", self._extract_pages_with_metadata)
        workflow.add_node("enhanced_chunking", self._enhanced_semantic_chunking)
        workflow.add_node("build_vector_store", self._build_enhanced_vector_store)
        
        # Add nodes for query processing
        workflow.add_node("analyze_query_complexity", self._analyze_query_complexity)
        workflow.add_node("temporal_retrieval", self._enhanced_temporal_retrieval)
        workflow.add_node("cfa_analysis", self._chartered_financial_analyst)
        workflow.add_node("generate_response", self._generate_enhanced_response)
        
        # Add edges for document processing
        workflow.add_edge("extract_files", "extract_pages")
        workflow.add_edge("extract_pages", "enhanced_chunking")
        workflow.add_edge("enhanced_chunking", "build_vector_store")
        workflow.add_edge("build_vector_store", END)
        
        # Add conditional edges for query processing
        workflow.add_edge("analyze_query_complexity", "temporal_retrieval")
        workflow.add_edge("temporal_retrieval", "cfa_analysis")
        workflow.add_edge("cfa_analysis", "generate_response")
        workflow.add_edge("generate_response", END)
        
        workflow.set_entry_point("extract_files")
        
        return workflow.compile()

    def _extract_pdf_files(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Extract list of PDF files with enhanced metadata"""
        documents_folder = os.path.join(os.getcwd(), "Documents_Sample")
        pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files")
        
        state["pdf_files"] = pdf_files
        return state

    def _extract_pages_with_metadata(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Extract pages with metadata for better chunking"""
        all_pages = []
        
        for pdf_path in state["pdf_files"]:
            logging.info(f"Extracting pages from: {pdf_path}")
            document_name = os.path.basename(pdf_path).replace('.pdf', '')
            
            try:
                with open(pdf_path, "rb") as pdf:
                    pdf_reader = PdfReader(pdf)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Extract temporal info from filename
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
                            all_pages.append(page_data)
                            
            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
                continue
        
        state["extracted_pages"] = all_pages
        return state

    def _enhanced_semantic_chunking(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Enhanced semantic chunking with page and table awareness"""
        all_chunks = []
        
        for page_data in state["extracted_pages"]:
            self.rate_limiter.wait_if_needed()
            
            chunks = self._process_page_with_semantic_chunking(page_data)
            all_chunks.extend(chunks)
            
            # Save chunks for this document
            self._save_document_chunks(page_data['document_name'], chunks)
        
        state["temporal_chunks"] = all_chunks
        return state

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
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                chunk_data = json.loads(json_match.group())
                
                chunks = []
                for i, item in enumerate(chunk_data):
                    chunk = EnhancedTemporalChunk(
                        text=item.get("text", ""),
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
        # Simple strategy: chunk by sections while keeping page context
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

    def _save_document_chunks(self, document_name: str, chunks: List[EnhancedTemporalChunk]):
        """Save chunks for a specific document"""
        filename = os.path.join(DOCUMENT_CHUNKS_DIR, f"{document_name}_chunks.json")
        chunk_data = [chunk.to_dict() for chunk in chunks]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(chunks)} chunks for document {document_name}")

    def _build_enhanced_vector_store(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Build enhanced vector store with rich metadata"""
        try:
            chunks = state["temporal_chunks"]
            
            texts = []
            metadatas = []
            
            for chunk in chunks:
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
            self._save_all_temporal_chunks(chunks)
            
            logging.info(f"Built enhanced vector store with {len(chunks)} chunks")
            
        except Exception as e:
            logging.error(f"Error building vector store: {e}")
            state["error"] = str(e)
        
        return state

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

    # Query processing methods
    def _analyze_query_complexity(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Analyze query complexity to determine processing approach"""
        query = state["query"]
        
        complexity_analysis_prompt = f"""
        Analyze this financial query to determine its complexity and processing requirements:
        
        Query: {query}
        
        Classify the complexity level and identify requirements:
        
        1. **complexity_level**:
           - 'simple': Basic data retrieval (specific metrics, single period)
           - 'moderate': Some analysis required (comparisons, trends)
           - 'complex': Deep analysis needed (root cause, strategic insights, multi-dimensional analysis)
        
        2. **analysis_type**:
           - 'data_retrieval': Just need to find and present data
           - 'comparative_analysis': Compare across periods/units
           - 'trend_analysis': Identify patterns and trends
           - 'root_cause_analysis': Deep dive into why something happened
           - 'strategic_analysis': Business implications and recommendations
        
        3. **requires_cfa**: True if needs chartered financial analyst expertise
        
        4. **temporal_scope**: Time periods involved
        5. **business_focus**: Specific business units or metrics
        
        Return JSON format.
        """
        
        try:
            self.rate_limiter.wait_if_needed()
            response = self.llm.invoke(complexity_analysis_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                intent = json.loads(json_match.group())
                state["query_intent"] = intent
                state["query_complexity"] = intent.get("complexity_level", "moderate")
            else:
                # Fallback analysis
                state["query_intent"] = self._fallback_query_analysis(query)
                state["query_complexity"] = "moderate"
                
        except Exception as e:
            logging.error(f"Error analyzing query complexity: {e}")
            state["query_intent"] = self._fallback_query_analysis(query)
            state["query_complexity"] = "moderate"
        
        return state

    def _fallback_query_analysis(self, query: str) -> Dict:
        """Fallback query analysis"""
        complexity_indicators = ['why', 'cause', 'reason', 'analyze', 'deep dive', 'insight', 'strategy']
        has_complexity = any(indicator in query.lower() for indicator in complexity_indicators)
        
        return {
            "complexity_level": "complex" if has_complexity else "moderate",
            "analysis_type": "root_cause_analysis" if has_complexity else "comparative_analysis",
            "requires_cfa": has_complexity,
            "temporal_scope": ["recent"],
            "business_focus": []
        }

    def _enhanced_temporal_retrieval(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Enhanced retrieval with complexity-aware ranking"""
        query = state["query"]
        intent = state["query_intent"]
        
        try:
            # Load vector store if not loaded
            if self.vector_store is None:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            # Enhanced query construction
            enhanced_query = self._construct_enhanced_query(query, intent)
            
            # Retrieve more documents for complex queries
            k = 30 if intent.get("complexity_level") == "complex" else 20
            docs_with_scores = self.vector_store.similarity_search_with_score(enhanced_query, k=k)
            
            # Advanced filtering and ranking
            filtered_docs = self._advanced_filter_and_rank(docs_with_scores, intent)
            
            # Convert to TemporalChunks
            retrieved_chunks = []
            for doc, score in filtered_docs[:15]:  # Top 15 for complex queries
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                original_chunk = self._find_chunk_by_id(chunk_id)
                if original_chunk:
                    retrieved_chunks.append(original_chunk)
            
            state["retrieved_chunks"] = retrieved_chunks
            
        except Exception as e:
            logging.error(f"Error in enhanced retrieval: {e}")
            state["error"] = str(e)
        
        return state

    def _construct_enhanced_query(self, query: str, intent: Dict) -> str:
        """Construct enhanced query based on intent"""
        enhanced_parts = [query]
        
        if intent.get("business_focus"):
            enhanced_parts.append(f"Business focus: {' '.join(intent['business_focus'])}")
        
        if intent.get("temporal_scope"):
            enhanced_parts.append(f"Time period: {' '.join(intent['temporal_scope'])}")
        
        if intent.get("analysis_type") == "root_cause_analysis":
            enhanced_parts.append("analysis explanation reasoning factors impact")
        
        return " ".join(enhanced_parts)

    def _advanced_filter_and_rank(self, docs_with_scores, intent):
        """Advanced filtering and ranking based on intent"""
        filtered = []
        
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            relevance_boost = 0
            
            # Boost for complexity match
            if intent.get("complexity_level") == metadata.get("complexity_level"):
                relevance_boost += 0.3
            
            # Boost for business unit relevance
            if intent.get("business_focus"):
                doc_units = metadata.get("business_units", [])
                if any(unit in doc_units for unit in intent["business_focus"]):
                    relevance_boost += 0.3
            
            # Boost for comparative analysis if query needs it
            if intent.get("analysis_type") in ["comparative_analysis", "trend_analysis"]:
                if metadata.get("has_comparisons", False):
                    relevance_boost += 0.2
            
            # Boost for financial tables in complex queries
            if intent.get("complexity_level") == "complex" and metadata.get("chunk_type") == "table":
                relevance_boost += 0.2
            
            adjusted_score = score - relevance_boost
            filtered.append((doc, adjusted_score))
        
        return sorted(filtered, key=lambda x: x[1])

    def _find_chunk_by_id(self, chunk_id: str) -> Optional[EnhancedTemporalChunk]:
        """Find chunk by ID from loaded chunks"""
        for chunk in self.temporal_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def _chartered_financial_analyst(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Chartered Financial Analyst for complex analysis"""
        query = state["query"]
        chunks = state["retrieved_chunks"]
        intent = state["query_intent"]
        
        if not intent.get("requires_cfa", False) and intent.get("complexity_level") != "complex":
            state["cfa_analysis"] = None
            state["cfa_reasoning"] = None
            return state
        
        # Prepare comprehensive context
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
        
        cfa_prompt = f"""
        You are a senior Chartered Financial Analyst (CFA) with 15+ years of experience in financial analysis, 
        corporate finance, and strategic consulting. You specialize in SaaS company analysis and travel technology sector.
        
        **CRITICAL INSTRUCTION**: Use <thinking> tags to show your analytical reasoning process step by step. 
        This thinking should be visible to demonstrate your analytical methodology.
        
        Query: {query}
        Intent Analysis: {json.dumps(intent, indent=2)}
        
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
            
            # Extract thinking and analysis
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            thinking = thinking_match.group(1).strip() if thinking_match else ""
            
            # Remove thinking from main analysis
            analysis = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL).strip()
            
            state["cfa_reasoning"] = thinking
            state["cfa_analysis"] = analysis
            
        except Exception as e:
            logging.error(f"Error in CFA analysis: {e}")
            state["cfa_analysis"] = None
            state["cfa_reasoning"] = None
        
        return state

    def _generate_enhanced_response(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """Generate enhanced response combining retrieval and CFA analysis"""
        query = state["query"]
        chunks = state["retrieved_chunks"]
        intent = state["query_intent"]
        cfa_analysis = state.get("cfa_analysis")
        
        # Prepare context
        context_parts = []
        for chunk in chunks[:10]:  # Top 10 chunks
            context_parts.append(f"""
            --- {chunk.month} {chunk.year} | {chunk.section_title} ---
            {chunk.text}
            """)
        
        context = "\n\n".join(context_parts)
        
        # Choose prompt based on complexity
        if intent.get("complexity_level") == "complex" and cfa_analysis:
            prompt_template = self._get_complex_analysis_prompt()
            response_input = {
                "context": context,
                "question": query,
                "cfa_analysis": cfa_analysis,
                "intent": json.dumps(intent, indent=2)
            }
        else:
            prompt_template = self._get_standard_analysis_prompt()
            response_input = {
                "context": context,
                "question": query,
                "intent": json.dumps(intent, indent=2)
            }
        
        try:
            self.rate_limiter.wait_if_needed()
            response = self.llm.invoke(prompt_template.format(**response_input))
            content = response.content if hasattr(response, 'content') else str(response)
            
            state["final_response"] = content
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            state["error"] = str(e)
        
        return state

    def _get_complex_analysis_prompt(self) -> str:
        """Prompt for complex analysis with CFA insights"""
        return """
        You are a senior financial analyst providing insights based on comprehensive analysis.
        
        A Chartered Financial Analyst has provided the following expert analysis:
        
        **CFA Expert Analysis:**
        {cfa_analysis}
        
        **User Query:** {query}
        **Analysis Intent:** {intent}
        
        **Supporting Financial Data:**
        {context}
        
        Synthesize the CFA analysis with the supporting data to provide a comprehensive response that:
        
        1. **Direct Answer**: Address the specific question clearly
        2. **Key Insights**: Highlight the most important findings
        3. **Financial Context**: Provide relevant background and benchmarking
        4. **Root Cause Analysis**: Explain the 'why' behind the numbers
        5. **Business Implications**: What this means for the company
        6. **Actionable Recommendations**: Specific next steps
        
        Structure your response clearly with headers and bullet points for readability.
        Include specific numbers and percentages where available.
        
        **Comprehensive Analysis:**
        """

    def _get_standard_analysis_prompt(self) -> str:
        """Prompt for standard analysis"""
        return """
        You are an expert financial analyst with detailed knowledge of RateGain Travel Technologies.
        
        **User Query:** {question}
        **Analysis Intent:** {intent}
        
        **Financial Data Context:**
        {context}
        
        Provide a comprehensive analysis that includes:
        
        1. **Direct Answer**: Address the specific question
        2. **Key Financial Metrics**: Present relevant numbers clearly
        3. **Trends and Patterns**: Identify important trends
        4. **Comparative Analysis**: Context vs. budget, prior periods
        5. **Business Insights**: What the data reveals about performance
        
        Format your response with clear headers and use specific numbers where available.
        If any information is missing, clearly state what is not available.
        
        **Financial Analysis:**
        """

    # Main interface methods
    def build_knowledge_base(self):
        """Build the enhanced knowledge base"""
        initial_state = FinancialAnalysisState(
            pdf_files=[],
            current_file=None,
            current_page=None,
            extracted_pages=[],
            temporal_chunks=[],
            query=None,
            query_intent=None,
            query_complexity=None,
            retrieved_chunks=[],
            cfa_analysis=None,
            cfa_reasoning=None,
            final_response=None,
            error=None
        )
        
        result = self.workflow.invoke(initial_state)
        
        if result.get("error"):
            logging.error(f"Error building knowledge base: {result['error']}")
            return False
        
        self.temporal_chunks = result["temporal_chunks"]
        return True

    def query(self, user_question: str) -> Dict[str, Any]:
        """Process a user query and return comprehensive results"""
        # Load chunks if not loaded
        if not self.temporal_chunks:
            self.temporal_chunks = self._load_temporal_chunks()
        
        query_state = FinancialAnalysisState(
            pdf_files=[],
            current_file=None,
            current_page=None,
            extracted_pages=[],
            temporal_chunks=self.temporal_chunks,
            query=user_question,
            query_intent=None,
            query_complexity=None,
            retrieved_chunks=[],
            cfa_analysis=None,
            cfa_reasoning=None,
            final_response=None,
            error=None
        )
        
        # Create query-specific workflow
        query_workflow = StateGraph(FinancialAnalysisState)
        query_workflow.add_node("analyze_query_complexity", self._analyze_query_complexity)
        query_workflow.add_node("temporal_retrieval", self._enhanced_temporal_retrieval)
        query_workflow.add_node("cfa_analysis", self._chartered_financial_analyst)
        query_workflow.add_node("generate_response", self._generate_enhanced_response)
        
        query_workflow.add_edge("analyze_query_complexity", "temporal_retrieval")
        query_workflow.add_edge("temporal_retrieval", "cfa_analysis")
        query_workflow.add_edge("cfa_analysis", "generate_response")
        query_workflow.add_edge("generate_response", END)
        
        query_workflow.set_entry_point("analyze_query_complexity")
        
        compiled_workflow = query_workflow.compile()
        result = compiled_workflow.invoke(query_state)
        
        return {
            "response": result.get("final_response", "No response generated"),
            "cfa_reasoning": result.get("cfa_reasoning"),
            "cfa_analysis": result.get("cfa_analysis"),
            "query_complexity": result.get("query_complexity"),
            "error": result.get("error")
        }

# Enhanced Streamlit Application
def main():
    st.set_page_config(
        page_title="Enhanced Financial RAG with CFA", 
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏢 Enhanced Financial Reports Analysis with CFA Intelligence")
    st.markdown("*Advanced Temporal-aware RAG with Chartered Financial Analyst for RateGain Financial Data*")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedFinancialRAG()
    
    rag_system = st.session_state.rag_system
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Build knowledge base button
        if st.button("🔨 Build Enhanced Knowledge Base", type="primary"):
            with st.spinner("Building enhanced knowledge base with page-aware chunking..."):
                success = rag_system.build_knowledge_base()
                if success:
                    st.success("✅ Enhanced knowledge base built successfully!")
                    st.info("📄 Semantic chunking with page and table awareness completed")
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
                # Document distribution
                doc_counts[chunk.document_name] = doc_counts.get(chunk.document_name, 0) + 1
                # Complexity distribution
                complexity_counts[chunk.complexity_level] = complexity_counts.get(chunk.complexity_level, 0) + 1
                # Chunk type distribution
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
        
        # CFA Intelligence status
        st.subheader("🎓 CFA Intelligence")
        st.info("Chartered Financial Analyst agent ready for complex queries")
        st.write("**Capabilities:**")
        st.write("• Root cause analysis")
        st.write("• Strategic financial insights")
        st.write("• Risk assessment")
        st.write("• Business recommendations")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "I'm ready to provide advanced financial analysis! Ask me complex questions and I'll engage our CFA intelligence for deep insights."}
            ]
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm ready to provide advanced financial analysis! Ask me complex questions and I'll engage our CFA intelligence for deep insights."}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "cfa_reasoning" in message:
                # Display CFA reasoning if available
                if message.get("cfa_reasoning"):
                    with st.expander("🧠 CFA Analytical Reasoning", expanded=False):
                        st.write(message["cfa_reasoning"])
                
                st.write(message["content"])
                
                if message.get("query_complexity"):
                    st.caption(f"Query Complexity: {message['query_complexity'].title()}")
            else:
                st.write(message["content"])
    
    # Enhanced chat input
    if prompt := st.chat_input("Ask complex financial questions for CFA-level analysis..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate enhanced response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with CFA intelligence..."):
                result = rag_system.query(prompt)
                
                # Display CFA reasoning if available
                if result.get("cfa_reasoning"):
                    with st.expander("🧠 CFA Analytical Reasoning", expanded=True):
                        st.write(result["cfa_reasoning"])
                
                # Display main response
                response = result["response"]
                st.write(response)
                
                # Display complexity level
                if result.get("query_complexity"):
                    st.caption(f"Query Complexity: {result['query_complexity'].title()}")
                
                # Handle errors
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
        
        # Add enhanced assistant message
        assistant_message = {
            "role": "assistant", 
            "content": result["response"],
            "cfa_reasoning": result.get("cfa_reasoning"),
            "query_complexity": result.get("query_complexity")
        }
        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()