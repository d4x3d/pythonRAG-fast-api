import os
import json
import tempfile
import psycopg2
from typing import Dict, List, Union
from dotenv import load_dotenv
from termcolor import colored, cprint
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector
from supabase import create_client, Client
import tiktoken
from dataclasses import dataclass
import hashlib
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Load environment variables
load_dotenv()

# Define state types
class GraphState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    next: Optional[str]

@dataclass
class ProcessedDocument:
    file_hash: str
    path: str
    total_tokens: int
    chunks: int

class PGVectorRAGPipeline:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.session_id = str(uuid.uuid4())  # Generate unique session ID
        self.selected_documents = set()  # Track selected documents
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.prompt = None
        self.temp_dir = tempfile.mkdtemp()
        self.supabase_client = None
        self.connection_string = f"postgresql://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:5432/{os.getenv('SUPABASE_DB_NAME')}"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.full_document = None
        self.processed_docs = {}
        self.messages = []  # Initialize messages list
        
        # Initialize the state graph for conversation management
        self.graph = StateGraph(GraphState)
        
        # First validate connection before proceeding
        if not self.validate_connection():
            raise Exception("Failed to validate database connection")
        
        # Initialize database tables
        self.initialize_tables()
        
        self.setup_pipeline()

    def validate_connection(self) -> bool:
        """Validate Supabase and database connections"""
        if self.verbose:
            cprint("\n🔄 Validating Supabase connection...", "cyan")
        
        try:
            # Validate environment variables
            required_vars = [
                "SUPABASE_URL",
                "SUPABASE_SERVICE_ROLE_KEY",
                "SUPABASE_DB_HOST",
                "SUPABASE_DB_NAME",
                "SUPABASE_DB_USER",
                "SUPABASE_DB_PASSWORD"
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

            # Initialize Supabase client
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            self.supabase_client = create_client(supabase_url, supabase_key)

            # Set the correct connection string format for pgvector
            self.connection_string = f"postgresql://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:5432/{os.getenv('SUPABASE_DB_NAME')}"
            
            # Test direct database connection
            conn = psycopg2.connect(
                host=os.getenv('SUPABASE_DB_HOST'),
                database=os.getenv('SUPABASE_DB_NAME'),
                user=os.getenv('SUPABASE_DB_USER'),
                password=os.getenv('SUPABASE_DB_PASSWORD')
            )
            
            # Test if pgvector extension is installed
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                has_vector = cur.fetchone()[0]
                if not has_vector:
                    raise ValueError("pgvector extension is not installed in the database")
                
                # Test if document_sections table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM information_schema.tables 
                        WHERE table_name = 'document_sections'
                    )
                """)
                has_table = cur.fetchone()[0]
                if not has_table:
                    cprint("\n⚠️ document_sections table not found. Creating it...", "yellow")
                    cur.execute("""
                        CREATE TABLE document_sections (
                            id bigserial primary key,
                            content text,
                            metadata jsonb,
                            embedding vector(384)
                        );
                        CREATE INDEX ON document_sections USING hnsw (embedding vector_cosine_ops);
                    """)
                    conn.commit()
                    cprint("✅ Created document_sections table and index", "green")

            conn.close()
            
            if self.verbose:
                cprint("✅ Supabase connection validated successfully", "green")
            return True

        except Exception as e:
            cprint(f"❌ Connection validation failed: {str(e)}", "red")
            return False

    def setup_pipeline(self):
        """Initialize RAG components with Supabase pgvector"""
        if self.verbose:
            cprint("\n🔧 Initializing RAG pipeline...", "cyan")
        
        try:
            # Initialize models
            self.llm = ChatGroq(model="llama-3.3-70b-versatile")
            self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
            
            # Initialize pgvector store
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name="document_sections",
                connection=self.connection_string,
                pre_delete_collection=False
            )
            
            # Enhanced prompt template with clear instructions about document reliance
            self.prompt = PromptTemplate.from_template("""
            You are a helpful AI assistant. Your primary source of information is the provided document context.
            
            Latest Exchange from Conversation:
            {chat_history}
            
            Document Context:
            {context}
            
            Current Question: {question}
            
            Instructions:
            1. Base your answer primarily on the provided document context
            2. If the document context doesn't contain information relevant to the question, clearly state that
            3. Use the latest exchange for conversation continuity, but prioritize document information
            4. Be direct and specific in your response
            5. If you need to reference information outside the document, clearly indicate this
            
            Please provide your answer:""")
            
            if self.verbose:
                cprint("✅ RAG pipeline initialized successfully", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize RAG pipeline: {str(e)}", "red")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_document_processed(self, file_path: str) -> bool:
        """Check if document is already processed"""
        try:
            file_hash = self.get_file_hash(file_path)
            
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                # First check the documents table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM documents 
                        WHERE file_hash = %s
                    )
                """, (file_hash,))
                exists = cur.fetchone()[0]
            
            conn.close()
            return exists
            
        except Exception as e:
            cprint(f"❌ Error checking document status: {str(e)}", "red")
            return False

    def ensure_chat_session(self):
        """Ensure a chat session exists, creating one if necessary"""
        try:
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                # Check if session exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM chat_sessions 
                        WHERE session_id = %s
                    )
                """, (self.session_id,))
                session_exists = cur.fetchone()[0]
                
                if not session_exists:
                    # Create new session
                    cur.execute("""
                        INSERT INTO chat_sessions (session_id)
                        VALUES (%s)
                        ON CONFLICT (session_id) DO NOTHING
                        RETURNING session_id
                    """, (self.session_id,))
                    conn.commit()
                    if self.verbose:
                        cprint(f"✅ Created new chat session: {self.session_id}", "green")
            
            conn.close()
            return True
        except Exception as e:
            cprint(f"❌ Error ensuring chat session: {str(e)}", "red")
            raise

    def process_pdf(self, pdf_path: str):
        """Process PDF with automatic session management"""
        if self.verbose:
            cprint(f"\n📑 Processing PDF: {pdf_path}", "cyan")

        try:
            # Ensure we have a valid session before processing
            self.ensure_chat_session()
            
            file_hash = self.get_file_hash(pdf_path)
            
            # Check if already processed
            if self.is_document_processed(pdf_path):
                cprint("📘 Document already processed, adding to session...", "yellow")
                self.add_document_to_session(pdf_path)
                return None

            # Load and process the PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Clean and join text
            full_text = "\n".join(page.page_content.replace('\x00', '') for page in pages)
            
            # Count tokens
            current_tokens = self.count_tokens(full_text)
            max_tokens = 120000  # Token limit
            
            if current_tokens > max_tokens:
                cprint(f"\n⚠️ Document was truncated to fit token limit", "yellow")
                # Truncate text to fit token limit while maintaining integrity
                while current_tokens > max_tokens and full_text:
                    full_text = full_text[:int(len(full_text) * 0.9)]  # Reduce by 10%
                    current_tokens = self.count_tokens(full_text)
            
            if self.verbose:
                cprint(f"Document tokens: {current_tokens}/{max_tokens}", "cyan")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            chunks = [
                Document(
                    page_content=chunk,
                    metadata={
                        "file_hash": file_hash,
                        "source": pdf_path,
                        "chunk_id": i
                    }
                )
                for i, chunk in enumerate(text_splitter.split_text(full_text))
            ]
            
            # Store processed document info
            self.processed_docs[file_hash] = ProcessedDocument(
                file_hash=file_hash,
                path=pdf_path,
                total_tokens=current_tokens,
                chunks=len(chunks)
            )

            # Add chunks to vector store
            self.vector_store.add_documents(chunks)

            if self.verbose:
                cprint("✅ Successfully stored documents in pgvector", "green")

            # Store document info in documents table
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (file_hash, filename, total_chunks, total_tokens)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (file_hash) DO NOTHING
                """, (file_hash, pdf_path, len(chunks), current_tokens))
                conn.commit()
            conn.close()

            # Add document to current session
            self.add_document_to_session(pdf_path)

            return chunks

        except Exception as e:
            cprint(f"❌ Error processing PDF: {str(e)}", "red")
            raise

    def create_chat_session(self):
        """Create a new chat session"""
        try:
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (session_id)
                    VALUES (%s)
                """, (self.session_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            cprint(f"❌ Error creating chat session: {str(e)}", "red")
            raise

    def add_document_to_session(self, pdf_path: str):
        """Add document to current chat session"""
        try:
            file_hash = self.get_file_hash(pdf_path)
            
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_session_documents (session_id, file_hash)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id, file_hash) DO NOTHING
                """, (self.session_id, file_hash))
                conn.commit()
            conn.close()
            
            self.selected_documents.add(pdf_path)
            if self.verbose:
                cprint(f"✅ Added document to chat session: {pdf_path}", "green")
                
        except Exception as e:
            cprint(f"❌ Error adding document to session: {str(e)}", "red")
            raise

    def start_chat_with_documents(self, file_paths: List[str]):
        """Initialize a chat session with specific documents"""
        try:
            # Create new session
            self.create_chat_session()
            
            # Process and add each document
            for file_path in file_paths:
                self.process_pdf(file_path)
            
            if self.verbose:
                cprint(f"✅ Chat session initialized with {len(file_paths)} documents", "green")
            
        except Exception as e:
            cprint(f"❌ Error starting chat session: {str(e)}", "red")
            raise

    def retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant documents from vector store"""
        if self.verbose:
            cprint("\n🔍 Retrieving relevant documents...", "cyan")
            cprint(f"📊 Session has {len(self.selected_documents)} documents", "cyan")
        
        try:
            # Get documents for current session
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.file_hash 
                    FROM chat_session_documents csd
                    JOIN documents d ON d.file_hash = csd.file_hash
                    WHERE csd.session_id = %s
                """, (self.session_id,))
                session_docs = [row[0] for row in cur.fetchall()]
            conn.close()

            # Perform similarity search with metadata filter
            docs = self.vector_store.similarity_search(
                question,
                k=10,  # Retrieve top 5 most relevant chunks
                filter={"file_hash": {"$in": session_docs}} if session_docs else None
            )
            
            if self.verbose:
                cprint(f"✅ Retrieved {len(docs)} relevant qewrys", "green")
            
            return docs
            
        except Exception as e:
            cprint(f"❌ Error retrieving documents: {str(e)}", "red")
            raise

    def add_message(self, message: Union[HumanMessage, AIMessage]):
        """Add a message to the conversation history"""
        self.messages.append(message)

    def get_chat_history(self) -> str:
        """Get formatted chat history"""
        history = []
        for msg in self.messages:
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            history.append(f"{role}: {msg.content}")
        return "\n".join(history)

    def get_structured_chat_history(self) -> str:
        """Get formatted chat history with clear Q&A structure"""
        if not self.messages:
            return "No previous conversation."
            
        structured_history = []
        for i in range(0, len(self.messages), 2):
            if i + 1 < len(self.messages):
                question = self.messages[i].content
                answer = self.messages[i + 1].content
                structured_history.append(f"Previous Question: {question}\nPrevious Answer: {answer}\n")
        
        return "\n".join(structured_history)

    def get_latest_exchange(self) -> str:
        """Get only the most recent Q&A exchange"""
        if len(self.messages) < 2:
            return ""
            
        # Get the last question and answer
        last_question = self.messages[-2].content
        last_answer = self.messages[-1].content
        return f"Previous Question: {last_question}\nPrevious Answer: {last_answer}"

    def analyze_query(self, question: str) -> Dict[str, any]:
        """Analyze the query in context of structured chat history"""
        try:
            analysis_prompt = PromptTemplate.from_template("""
            Analyze this query in the context of the previous conversation.
            
            Conversation History:
            {chat_history}
            
            Current Question: {question}
            
            Analyze the query and provide key information about:
            1. Main topic
            2. Related topics
            3. References to previous conversation
            4. Key search terms
            
            Response:""")
            
            analysis_chain = (
                analysis_prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            # Get analysis
            analysis = analysis_chain.invoke({
                "chat_history": self.get_structured_chat_history(),
                "question": question
            })
            
            # Parse the response into a structured format
            return {
                "main_topic": question,
                "related_topics": [],
                "references_previous_qa": "previous" in analysis.lower(),
                "search_keywords": question.split()
            }
            
        except Exception as e:
            cprint(f"⚠️ Query analysis failed: {str(e)}", "yellow")
            return {
                "main_topic": question,
                "related_topics": [],
                "references_previous_qa": False,
                "search_keywords": question.split()
            }

    def generate(self, question: str, context_docs: List[Document]):
        """Generate answer with enhanced context awareness and structured history"""
        if self.verbose:
            cprint("\n🤖 Generating answer...", "cyan")
        
        try:
            # Add the new question to chat history
            self.add_message(HumanMessage(content=question))
            
            # Get only the latest exchange for context
            latest_exchange = self.get_latest_exchange()
            
            # Prepare context from documents
            context = "\n\n".join(doc.page_content for doc in context_docs)
            
            # Create chain with enhanced context
            chain = (
                {
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                    "chat_history": lambda _: latest_exchange
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response with full context
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Add the response to chat history
            self.add_message(AIMessage(content=response))
            
            if self.verbose:
                cprint("✅ Generated response", "green")
            
            return response
            
        except Exception as e:
            cprint(f"❌ Error generating answer: {str(e)}", "red")
            raise

    def initialize_tables(self):
        """Initialize necessary database tables if they don't exist"""
        try:
            conn = psycopg2.connect(self.connection_string)
            with conn.cursor() as cur:
                # Create documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        file_hash TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        total_chunks INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create chat sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create chat session documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_session_documents (
                        session_id TEXT REFERENCES chat_sessions(session_id),
                        file_hash TEXT REFERENCES documents(file_hash),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (session_id, file_hash)
                    );
                """)
                
                conn.commit()
                
            conn.close()
            if self.verbose:
                cprint("✅ Database tables initialized", "green")
                
        except Exception as e:
            cprint(f"❌ Error initializing tables: {str(e)}", "red")
            raise

def main():
    try:
        # Initialize the pipeline
        rag = PGVectorRAGPipeline()
        
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
            answer = rag.generate(question, relevant_docs)
            
            # Display answer
            print("\n🤖 Answer:")
            print(answer)

    except Exception as e:
        cprint(f"\n❌ An error occurred: {str(e)}", "red")
        raise

if __name__ == "__main__":
    main()
