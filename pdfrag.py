import os
import tempfile
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from termcolor import colored, cprint
import tiktoken

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["NOMIC_API_KEY"] = os.getenv("NOMIC_API_KEY")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class RAGPipeline:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.prompt = None
        self.temp_dir = tempfile.mkdtemp()
        self.setup_pipeline()

    def process_pdf(self, pdf_path: str, verbose=True):
        """Process a PDF file and return chunks with verification steps"""
        if verbose:
            print(colored(f"\nProcessing PDF: {pdf_path}", "cyan"))
        
        # Load PDF and verify content
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        if verbose:
            print(colored(f"Successfully loaded PDF: {len(docs)} pages found", "green"))
        
        if not docs:
            raise ValueError("No content extracted from PDF")
        
        # Count total tokens in the document
        total_tokens = sum(count_tokens(doc.page_content) for doc in docs)
        if verbose:
            print(colored(f"Total tokens in document: {total_tokens:,}", "yellow"))
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        all_splits = text_splitter.split_documents(docs)
        
        if verbose:
            print(colored(f"Created {len(all_splits)} text chunks", "green"))
            sample_chunk = all_splits[0].page_content[:200]
            print(colored("Sample chunk:", "yellow"))
            print(colored(sample_chunk + "...", "white"))
            print(colored(f"Sample chunk tokens: {count_tokens(sample_chunk)}", "yellow"))
        
        return all_splits

    def setup_pipeline(self):
        """Initialize RAG components with verification"""
        if self.verbose:
            cprint("\n🔧 Initializing RAG pipeline...", "cyan")
        
        try:
            # Initialize models
            self.llm = ChatGroq(model="llama-3.3-70b-versatile")
            self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
            self.vector_store = InMemoryVectorStore(self.embeddings)
            
            # Load RAG prompt
            self.prompt = hub.pull("rlm/rag-prompt")
            
            if self.verbose:
                cprint("✅ RAG pipeline initialized successfully", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize RAG pipeline: {str(e)}", "red")
            raise

    def retrieve(self, state: State):
        """Retrieve documents with visual feedback"""
        cprint("\n🔍 Retrieving relevant information...", "cyan")
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        cprint("✅ Retrieved relevant context", "green")
        return {"context": retrieved_docs}

    def generate(self, state: State):
        """Generate answer with visual feedback"""
        cprint("\n🤖 Generating response...", "cyan")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        cprint("✅ Response generated", "green")
        return {"answer": response.content}

    def save_chunks_to_temp(self, chunks, prefix="chunk"):
        """Save chunks to temporary files"""
        temp_path = os.path.join(self.temp_dir, f"{prefix}_data.json")
        chunks_data = [
            {
                "content": chunk.page_content,
                "tokens": count_tokens(chunk.page_content)
            } for chunk in chunks
        ]
        with open(temp_path, 'w') as f:
            json.dump(chunks_data, f)
        return temp_path

def get_pdf_files():
    """Get all PDF files from the pdf directory"""
    pdf_dir = os.path.join(os.path.dirname(__file__), "pdf")
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    return [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

def main(pdf_path: str, question: str, verbose=True):
    """Enhanced main function with better visual feedback"""
    try:
        # Initialize with loading animation
        cprint("\n🚀 Initializing RAG Pipeline...", "cyan", attrs=['bold'])
        rag = RAGPipeline(verbose=verbose)
        
        # Process PDF
        chunks = rag.process_pdf(pdf_path)
        
        # Add to vector store with progress indicator
        cprint("\n📥 Adding documents to vector store...", "cyan")
        rag.vector_store.add_documents(documents=chunks)
        cprint("✅ Documents indexed successfully", "green")
        
        # Create graph
        graph_builder = StateGraph(State).add_sequence([rag.retrieve, rag.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        
        # Get response
        response = graph.invoke({
            "question": question,
            "context": []  # Context will be retrieved in the pipeline
        })
        
        # Display results
        cprint("\n📝 Results", "cyan", attrs=['bold'])
        print(colored("Question: ", "yellow") + question)
        print(colored("Answer: ", "green") + response["answer"])
        print(colored(f"Answer tokens: {count_tokens(response['answer'])}", "blue"))
        
        return response["answer"]
        
    except Exception as e:
        cprint(f"\n❌ Error: {str(e)}", "red")
        raise

if __name__ == "__main__":
    try:
        # ASCII art header
        header = """
╔════════════════════════════════════╗
║          RAG PDF Assistant         ║
╚════════════════════════════════════╝
        """
        print(colored(header, "cyan", attrs=['bold']))
        
        # Get PDF files
        pdf_files = get_pdf_files()
        
        if not pdf_files:
            cprint("❌ No PDF files found!", "red")
            cprint("📁 Please add PDF files to the 'pdf' folder", "yellow")
            exit(1)
        
        # Display available PDFs
        cprint("\n📚 Available PDF Files:", "cyan", attrs=['bold'])
        for i, pdf in enumerate(pdf_files, 1):
            print(colored(f"{i}. 📄 {pdf}", "white"))
        
        # Get user selection
        while True:
            try:
                selection = int(input(colored("\n📎 Select a PDF file (enter number): ", "yellow"))) - 1
                if 0 <= selection < len(pdf_files):
                    break
                cprint("❌ Invalid selection. Please try again.", "red")
            except ValueError:
                cprint("❌ Please enter a valid number.", "red")
        
        # Get question
        cprint("\n❓ Ask a Question", "cyan", attrs=['bold'])
        question = input(colored("Enter your question: ", "yellow")).strip()
        
        if len(question) < 5:
            cprint("❌ Question too short! Please be more specific.", "red")
            exit(1)
        
        # Process
        pdf_path = os.path.join(os.path.dirname(__file__), "pdf", pdf_files[selection])
        main(pdf_path, question, verbose=True)
        
    except KeyboardInterrupt:
        cprint("\n\n👋 Process interrupted by user", "yellow")
    except Exception as e:
        cprint(f"\n❌ Error: {str(e)}", "red")
