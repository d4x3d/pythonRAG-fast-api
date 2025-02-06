import os
import json
import tempfile
import psycopg2
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from termcolor import colored, cprint
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import together
from together import Together
from langchain_nomic import NomicEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector
from supabase import create_client, Client
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import TypedDict
import tiktoken
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables
load_dotenv()

# Set Together API key
TOGETHER_API_KEY = "f42eed6d2fd5ed6554b13ac970c9c7882e5d30dac7866f756e6b32695a86b966"
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float

class ChatHistory:
    def __init__(self, max_history: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str, timestamp: float):
        self.messages.append(ChatMessage(role=role, content=content, timestamp=timestamp))
        if len(self.messages) > self.max_history:
            self.messages.pop(0)

    def get_context(self, window_size: int = 3) -> str:
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in self.messages[-window_size:]
        ])

class SearchResult(TypedDict):
    document: Document
    keyword_score: float
    vector_score: float
    context_score: float
    total_score: float

class EnhancedPGVectorRAGPipeline:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.connection_string = f"postgresql://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:5432/{os.getenv('SUPABASE_DB_NAME')}"
        self.keywords_file = Path("document_keywords.json")
        self.chat_history = ChatHistory()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.last_api_call = 0
        self.min_delay = 1.1  # Minimum 1.1 seconds between API calls (< 1 QPS)
        self.setup_pipeline()

    def setup_pipeline(self):
        """Initialize enhanced RAG components"""
        if self.verbose:
            cprint("\n🔧 Initializing Enhanced RAG pipeline...", "cyan")
        
        try:
            # Common configuration for Together AI
            base_config = {
                "base_url": "https://api.together.xyz/v1",
                "api_key": TOGETHER_API_KEY,
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
            }
            
            self.query_understanding_llm = ChatOpenAI(
                **base_config,
                temperature=0.3,
                max_tokens=500
            )
            
            self.keyword_extraction_llm = ChatOpenAI(
                **base_config,
                temperature=0.2,
                max_tokens=300
            )
            
            self.response_llm = ChatOpenAI(
                **base_config,
                temperature=0.7,
                max_tokens=2000
            )
            
            self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
            
            # Initialize vector store
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name="document_sections",
                connection=self.connection_string,
                pre_delete_collection=False
            )
            
            # Enhanced prompts
            self.query_analysis_prompt = PromptTemplate.from_template("""
            Analyze the following question and provide key search terms.
            Consider the conversation context for better understanding.
            
            Question: {question}
            Context: {context}
            
            Return a JSON with:
            {
                "keywords": ["key1", "key2"],
                "entities": ["entity1", "entity2"],
                "concepts": ["concept1", "concept2"]
            }
            """)
            
            self.keyword_extraction_prompt = PromptTemplate.from_template("""
            Extract key information from this text segment. Return a JSON with:
            - main_topics: List of 3-5 main topics
            - technical_terms: List of technical terms
            - named_entities: List of named entities
            - key_concepts: List of key concepts
            
            Text: {text}
            
            JSON response:""")
            
            self.response_prompt = PromptTemplate.from_template("""
            Previous conversation context:
            {chat_history}
            
            Current question: {question}
            
            Relevant information:
            {context}
            
            Instructions:
            1. Consider the conversation history for context
            2. Use only the provided information to answer
            3. If uncertain, acknowledge limitations
            4. Maintain consistency with previous responses
            
            Answer:""")
            
            if self.verbose:
                cprint("✅ Enhanced RAG pipeline initialized successfully", "green")
                
        except Exception as e:
            cprint(f"❌ Failed to initialize pipeline: {str(e)}", "red")
            raise

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_api_call = time.time()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def _safe_llm_call(self, llm, prompt, **kwargs):
        """Safe LLM call with rate limiting"""
        self._wait_for_rate_limit()
        try:
            return llm.invoke(prompt, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e):
                time.sleep(2)  # Additional delay on rate limit
                raise
            raise

    def process_chunk(self, chunk: Document) -> Dict:
        """Process a single document chunk with rate limiting"""
        try:
            # Extract keywords and metadata with rate limiting
            prompt = self.keyword_extraction_prompt.format(text=chunk.page_content)
            result = self._safe_llm_call(self.keyword_extraction_llm, prompt)
            
            try:
                parsed_result = json.loads(result.content)
            except (json.JSONDecodeError, AttributeError):
                # Fallback to empty structure if parsing fails
                parsed_result = {
                    "main_topics": [],
                    "technical_terms": [],
                    "named_entities": [],
                    "key_concepts": []
                }
            
            # Calculate scores
            complexity_score = self._calculate_complexity(chunk.page_content)
            importance_score = self._calculate_importance(parsed_result)
            
            return {
                "keywords": parsed_result,
                "complexity": complexity_score,
                "importance": importance_score
            }
        except Exception as e:
            cprint(f"⚠️ Chunk processing error: {str(e)}", "yellow")
            return {}

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        tokens = self.tokenizer.encode(text)
        unique_tokens = len(set(tokens))
        return unique_tokens / len(tokens)

    def _calculate_importance(self, keyword_data: Dict) -> float:
        """Calculate content importance score"""
        total_terms = sum(len(v) for v in keyword_data.values())
        technical_weight = len(keyword_data.get('technical_terms', [])) * 1.5
        return (total_terms + technical_weight) / 10

    def process_pdf(self, pdf_path: str):
        """Enhanced PDF processing with parallel execution"""
        if self.verbose:
            cprint(f"\n📑 Processing PDF: {pdf_path}", "cyan")

        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Clean PDF content by removing null characters
            cleaned_documents = []
            for doc in documents:
                cleaned_content = doc.page_content.replace('\x00', ' ').strip()
                if cleaned_content:  # Only add non-empty documents
                    cleaned_documents.append(Document(
                        page_content=cleaned_content,
                        metadata=doc.metadata
                    ))
            
            # Enhanced text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=lambda x: len(self.tokenizer.encode(x)),
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_documents(cleaned_documents)
            
            if self.verbose:
                cprint(f"📄 Processing {len(chunks)} chunks in parallel...", "cyan")
            
            # Parallel processing of chunks
            with ThreadPoolExecutor() as executor:
                chunk_data = list(executor.map(self.process_chunk, chunks))
            
            # Enhance chunks with extracted data
            enhanced_chunks = []
            for chunk, data in zip(chunks, chunk_data):
                if data:  # Only include chunks with successful processing
                    chunk.metadata.update({
                        "keywords": data["keywords"],
                        "complexity": data["complexity"],
                        "importance": data["importance"],
                        "source": pdf_path
                    })
                    enhanced_chunks.append(chunk)
            
            if enhanced_chunks:
                # Store enhanced chunks
                self.vector_store.add_documents(enhanced_chunks)
                
                if self.verbose:
                    cprint(f"✅ Successfully processed {len(enhanced_chunks)} chunks", "green")
            else:
                cprint("⚠️ No valid chunks were processed", "yellow")
            
            return enhanced_chunks

        except Exception as e:
            cprint(f"❌ Error processing PDF: {str(e)}", "red")
            raise

    def retrieve(self, question: str, k: int = 5) -> List[SearchResult]:
        """Enhanced retrieval with context awareness"""
        try:
            # Analyze query with context
            chain = self.query_analysis_prompt | self.query_understanding_llm | StrOutputParser()
            query_analysis = json.loads(chain.invoke({
                "question": question,
                "context": self.chat_history.get_context()
            }))
            
            # Get vector similarity results
            vector_results = self.vector_store.similarity_search_with_score(question, k=k*2)
            
            scored_results: List[SearchResult] = []
            for doc, vector_score in vector_results:
                # Calculate multiple relevance scores
                keyword_score = self._calculate_keyword_match(query_analysis, doc.metadata["keywords"])
                context_score = self._calculate_context_relevance(doc, self.chat_history)
                
                # Combine scores with weights
                total_score = (
                    vector_score * 0.4 +
                    keyword_score * 0.3 +
                    context_score * 0.3
                )
                
                scored_results.append({
                    "document": doc,
                    "keyword_score": keyword_score,
                    "vector_score": vector_score,
                    "context_score": context_score,
                    "total_score": total_score
                })
            
            # Sort by total score and return top k
            return sorted(scored_results, key=lambda x: x["total_score"], reverse=True)[:k]
            
        except Exception as e:
            cprint(f"❌ Error in retrieval: {str(e)}", "red")
            raise

    def _calculate_keyword_match(self, query_analysis: Dict, doc_keywords: Dict) -> float:
        """Calculate keyword matching score"""
        query_terms = set()
        for key in ["keywords", "entities", "concepts"]:
            query_terms.update(query_analysis.get(key, []))
        
        doc_terms = set()
        for terms in doc_keywords.values():
            doc_terms.update(terms)
        
        intersection = len(query_terms.intersection(doc_terms))
        return intersection / (len(query_terms) + 1e-6)

    def _calculate_context_relevance(self, doc: Document, chat_history: ChatHistory) -> float:
        """Calculate relevance to conversation context"""
        recent_context = chat_history.get_context()
        if not recent_context:
            return 0.5
        
        context_embedding = self.embeddings.embed_query(recent_context)
        doc_embedding = self.embeddings.embed_query(doc.page_content)
        
        return float(np.dot(context_embedding, doc_embedding))

    def generate_response(self, question: str, context_docs: List[SearchResult]) -> str:
        """Generate contextually aware response"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"[Score: {result['total_score']:.2f}]\n{result['document'].page_content}"
                for result in context_docs
            ])
            
            # Generate response using conversation history
            chain = (
                {
                    "question": lambda x: x["question"],
                    "context": lambda x: x["context"],
                    "chat_history": lambda x: x["chat_history"]
                }
                | self.response_prompt
                | self.response_llm
                | StrOutputParser()
            )
            
            response = chain.invoke({
                "question": question,
                "context": context,
                "chat_history": self.chat_history.get_context()
            })
            
            # Update chat history
            self.chat_history.add_message("user", question, time.time())
            self.chat_history.add_message("assistant", response, time.time())
            
            return response
            
        except Exception as e:
            cprint(f"❌ Error generating response: {str(e)}", "red")
            raise

def main():
    try:
        # Initialize the pipeline
        rag = EnhancedPGVectorRAGPipeline()
        
        # List PDF files in the pdf directory
        pdf_dir = "pdf"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            cprint(f"\n📁 Created '{pdf_dir}' directory", "yellow")
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            cprint("❌ No PDF files found in the 'pdf' directory", "red")
            return
        
        # Display available PDFs
        cprint("\n📚 Available PDF files:", "cyan")
        for i, file in enumerate(pdf_files, 1):
            print(f"{i}. {file}")
        
        # Get user selection
        while True:
            try:
                selection = int(input("\nSelect a PDF file (enter number): ")) - 1
                if 0 <= selection < len(pdf_files):
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_pdf = os.path.join(pdf_dir, pdf_files[selection])
        
        # Process the PDF
        rag.process_pdf(selected_pdf)
        
        # Interactive Q&A loop
        while True:
            question = input("\n❓ Ask a Question (or type 'exit' to quit): ")
            
            if question.lower() == 'exit':
                break
            
            # Retrieve relevant context
            relevant_docs = rag.retrieve(question)
            
            # Generate answer
            answer = rag.generate_response(question, relevant_docs)
            
            # Display answer
            print("\n🤖 Answer:")
            print(answer)

    except Exception as e:
        cprint(f"\n❌ An error occurred: {str(e)}", "red")
        raise

if __name__ == "__main__":
    main()
