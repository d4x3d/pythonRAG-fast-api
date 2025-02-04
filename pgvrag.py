import os
import json
import tempfile
import psycopg2
from typing import Dict, List
from dotenv import load_dotenv
from termcolor import colored, cprint
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector  # Updated import
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Type alias for state
State = Dict[str, any]

def count_tokens(text: str) -> int:
    """Estimate token count in text"""
    return len(text.split()) * 1.3

class PGVectorRAGPipeline:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.prompt = None
        self.temp_dir = tempfile.mkdtemp()
        self.supabase_client = None
        self.connection_string = None
        
        # First validate connection before proceeding
        if not self.validate_supabase_connection():
            raise ConnectionError("Failed to establish Supabase connection")
        
        self.setup_pipeline()

    def validate_supabase_connection(self) -> bool:
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

            # Test Supabase connection
            response = self.supabase_client.table('document_sections').select("id").limit(1).execute()
            
            # Build and test database connection string
            self.connection_string = f"postgresql+psycopg2://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:5432/{os.getenv('SUPABASE_DB_NAME')}"
            
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
            self.llm = ChatGroq(model="llama3-8b-8192")
            self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
            
            # Initialize pgvector store with the new implementation
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name="document_sections",
                connection=self.connection_string,
                pre_delete_collection=False
            )
            
            # Load RAG prompt
            self.prompt = PromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}
            
            Question: {question}
            
            Answer:""")
            
            if self.verbose:
                cprint("✅ RAG pipeline initialized successfully", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize RAG pipeline: {str(e)}", "red")
            raise

    def process_pdf(self, pdf_path: str):
        """Process PDF and store in pgvector"""
        if self.verbose:
            cprint(f"\n📑 Processing PDF: {pdf_path}", "cyan")

        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if self.verbose:
                cprint(f"📄 Loaded {len(documents)} pages", "green")

            # Clean PDF content by removing null characters
            cleaned_documents = []
            for doc in documents:
                cleaned_content = doc.page_content.replace('\x00', '')
                cleaned_documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_documents(cleaned_documents)

            if self.verbose:
                cprint(f"🔨 Created {len(chunks)} chunks", "green")

            # Add chunks to pgvector
            self.vector_store.add_documents(chunks)

            if self.verbose:
                cprint("✅ Successfully stored documents in pgvector", "green")

            return chunks

        except Exception as e:
            cprint(f"❌ Error processing PDF: {str(e)}", "red")
            raise

    def retrieve(self, query: str, k: int = 10):
        """Retrieve relevant documents using pgvector"""
        if self.verbose:
            cprint("\n🔍 Retrieving relevant documents...", "cyan")
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            
            if self.verbose:
                cprint(f"✅ Retrieved {len(docs)} relevant documents", "green")
            
            return docs
        
        except Exception as e:
            cprint(f"❌ Error retrieving documents: {str(e)}", "red")
            raise

    def generate(self, question: str, context_docs: List[Document]):
        """Generate answer using retrieved context"""
        if self.verbose:
            cprint("\n🤖 Generating answer...", "cyan")
        
        try:
            context = "\n\n".join(doc.page_content for doc in context_docs)
            
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            response = chain.invoke({"context": context, "question": question})
            
            if self.verbose:
                cprint("✅ Generated response", "green")
            
            return response
            
        except Exception as e:
            cprint(f"❌ Error generating answer: {str(e)}", "red")
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
