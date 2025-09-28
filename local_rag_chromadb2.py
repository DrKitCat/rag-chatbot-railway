import os
import uuid
import hashlib
from datetime import datetime, timezone
from typing import List, Dict
from dotenv import load_dotenv

# PDF processing
from pypdf import PdfReader

# ChromaDB for local vector storage
import chromadb
from chromadb.config import Settings

# Sentence transformers for local embeddings
#from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # Define as None to avoid NameError
    print("SentenceTransformers library is not installed. Please install it to use local embeddings.")

# Azure OpenAI for chat (keep this for responses)
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage

# Config
from config import get_logger

load_dotenv()

# Initialize logging
logger = get_logger(__name__)

class LocalRAGChatbot:
    def __init__(self, collection_name: str = "pdf_documents"):
        """Initialize the local RAG chatbot with ChromaDB."""
        logger.info("Initializing Local RAG Chatbot...")
        
        # Check if sentence transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformer is required but not available")
            raise ImportError("SentenceTransformer library not available. Please install sentence-transformers.")
        
        # Initialize local embedding model (free)
        logger.info("Loading local embedding model...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB client (local)

        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            # self.chroma_client = chromadb.Client(chromadb.config.Settings(
            # chroma_db_impl="duckdb+parquet",
            # persist_directory="./chroma_db"))
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "PDF document chunks"}
            )
            logger.info("✅ ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise
        
        # Initialize Azure OpenAI for chat responses only
        try:
            self.chat_client = ChatCompletionsClient(
                endpoint=os.environ["INFERENCE_ENDPOINT"],
                credential=AzureKeyCredential(os.environ["AZURE_OPENAI_API_KEY"]),
                api_version="2024-02-01"
            )
            logger.info("✅ Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
            raise
        
        logger.info("Local RAG Chatbot initialized successfully")
    
    def extract_pdf_text_by_page(self, pdf_path: str) -> List[str]:
        """Extract text from PDF, returning list of pages."""
        try:
            reader = PdfReader(pdf_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text.strip())
            return pages
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end == text_len:
                break
            start = max(start + chunk_size - overlap, start + 1)
        
        return chunks
    
    def create_deterministic_id(self, parent_id: str, chunk_index: int, content: str) -> str:
        """Create consistent ID based on content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{parent_id}-chunk-{chunk_index}-{content_hash}"
    
    def process_pdf_file(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 100) -> bool:
        """Process a single PDF file and add to ChromaDB."""
        filename = os.path.basename(pdf_path)
        parent_id = os.path.splitext(filename)[0]
        
        # Check if already processed
        existing = self.collection.get(where={"parent_id": parent_id})
        if existing['documents']:
            logger.info(f"PDF {filename} already processed. Skipping.")
            return True
        
        logger.info(f"Processing PDF: {filename}")
        
        pages = self.extract_pdf_text_by_page(pdf_path)
        if not pages:
            logger.warning(f"No text extracted from {filename}")
            return False
        
        # Process all chunks from the PDF
        all_documents = []
        all_metadatas = []
        all_ids = []
        chunk_counter = 0
        
        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue
                
            chunks = self.chunk_text(page_text, chunk_size, overlap)
            
            for chunk in chunks:
                chunk_counter += 1
                doc_id = self.create_deterministic_id(parent_id, chunk_counter, chunk)
                
                metadata = {
                    "title": filename,
                    "parent_id": parent_id,
                    "page_number": page_num,
                    "chunk_id": f"{parent_id}-p{page_num}-c{chunk_counter}",
                    "source": "pdf",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                
                all_documents.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(doc_id)
        
        if not all_documents:
            logger.warning(f"No chunks created from {filename}")
            return False
        
        # Add to ChromaDB (this automatically generates embeddings)
        try:
            self.collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"Successfully added {len(all_documents)} chunks from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def process_pdf_folder(self, folder_path: str, chunk_size: int = 1000, overlap: int = 100):
        """Process all PDF files in a folder."""
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return
        
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            if self.process_pdf_file(pdf_path, chunk_size, overlap):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
        
        # Show total documents in collection
        count = self.collection.count()
        logger.info(f"Total documents in local database: {count}")
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents in ChromaDB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            for i in range(len(results['documents'][0])):
                doc = {
                    "content": results['documents'][0][i],
                    "title": results['metadatas'][0][i]['title'],
                    "page_number": results['metadatas'][0][i]['page_number'],
                    "chunk_id": results['metadatas'][0][i]['chunk_id'],
                    "distance": results['distances'][0][i],
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity score
                }
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def create_context(self, documents: List[Dict]) -> str:
        """Create context string from retrieved documents."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_part = f"""
Document {i}:
Title: {doc['title']}
Page: {doc['page_number']}
Content: {doc['content']}
---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context."""
        system_prompt = """You are a helpful AI assistant specializing in UK tax regulations, particularly HMRC's Corporate Intangibles Research and Development (CIRD) manual.
 

nstructions:
- Use the provided document context as your primary source of information
- If the context contains relevant information, provide a comprehensive answer based on that content
- You can synthesize information from multiple documents to give a complete explanation
- Cite which documents or pages you're referencing when possible
- If the provided context doesn't fully answer the question but contains related information, explain what you can based on the available context
- Only say you don't have information if the context is completely irrelevant to the question"""


        user_prompt = f"""Context from CIRD manual documents:
{context}

Question: {query}

Please provide a comprehensive answer basaed on the document context above."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ]
            
            response = self.chat_client.complete(
                messages=messages,
                model=os.environ["CHAT_MODEL"],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    def get_hmrc_url(self,document_title: str) -> str:
        """Generate HMRC URL from document filename."""
        import re
    
    # Extract CIRD code from filename
        match = re.search(r'cird(?:update)?(\d+)', document_title.lower())
        if match:
            cird_code = match.group(1)
            return f"https://www.gov.uk/hmrc-internal-manuals/corporate-intangibles-research-and-development-manual/cird{cird_code}"
    
        return "https://www.gov.uk/hmrc-internal-manuals/corporate-intangibles-research-and-development-manual"
    
    
    def chat(self, query: str, top_k: int = 5) -> Dict:
        """Main chat method - performs RAG pipeline."""
        logger.info(f"Processing query: {query}")
        
        # Search for relevant documents
        documents = self.search_documents(query, top_k)
        
        if not documents:
            return {
                "response": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "query": query
            }
        
        # Create context from documents
        context = self.create_context(documents)
        
        # Generate response
        response = self.generate_response(query, context)
        
        # Prepare response with sources
        sources = []
        for doc in documents:
            source = {
                "title": doc["title"],
                "page": doc["page_number"],
                "score": round(doc["score"], 3),
                "chunk_id": doc["chunk_id"],
                "url": self.get_hmrc_url(doc["title"])
            }
            sources.append(source)
        
        return {
            "response": response,
            "sources": sources,
            "query": query,
            "num_sources": len(documents)
        }
        

    def get_collection_info(self):
        """Get information about the current collection."""
        count = self.collection.count()
        logger.info(f"Collection contains {count} documents")
        return count

    def determine_query_type(self, query: str) -> str:
        """Determine if query needs RAG or can be answered generally."""
        rag_indicators = [
            "cird", "manual", "hmrc", "tax", "r&d", "sme", "relief",
            "expenditure", "corporate", "intangibles", "development"
        ]
        
        greeting_patterns = ["hello", "hi", "good morning", "thanks", "thank you", "how are you"]
        
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in rag_indicators):
            print(f"DEBUG: Routing to RAG - found indicator in: {query}")
            return "rag"
        elif any(greeting in query_lower for greeting in greeting_patterns):
            print(f"DEBUG: Routing to general - found greeting in: {query}")
            return "general"
        else:
            print(f"DEBUG: Routing to general (default) for: {query}")
            return "general"

    def chat_with_routing(self, query: str) -> Dict:
        """Route query to appropriate handler."""
        query_type = self.determine_query_type(query)
        
        if query_type == "rag":
            return self.chat(query)  # Your existing RAG pipeline
        else:
            return self.general_chat(query)  # Direct LLM for general queries

    def general_chat(self, query: str) -> Dict:
        """Handle general queries without RAG."""
        
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
        messages = [
        SystemMessage(content=f"""You are a helpful assistant. Be friendly and informative. 
        
        Current date and time: {current_time}
        
        If asked about the current date, time, or day, use this information."""),
        UserMessage(content=query)
        ]
        
        # messages = [
        #     SystemMessage(content="You are a helpful assistant. Be friendly and informative."),
        #     UserMessage(content=query)
        messages = [
        SystemMessage(content=f"""You are a helpful assistant. Be friendly and informative. 
        Today's date is {datetime.now().strftime('%B %d, %Y')}. 
        If asked about the current date or time, use this information."""),
        UserMessage(content=query)
        ]
        
        response = self.chat_client.complete(
            messages=messages,
            model=os.environ["CHAT_MODEL"]
        )
        
        return {
            "response": response.choices[0].message.content,
            "sources": [],
            "query": query,
            "type": "general"
        }

def interactive_chat():
    """Run interactive chat session."""
    try:
        chatbot = LocalRAGChatbot()
        
        print(f"\nCollection contains {chatbot.get_collection_info()} documents")
        print("\n" + "="*60)
        print("Local RAG Chatbot Ready!")
        print("Ask questions about your uploaded documents.")
        print("Type 'quit' or 'exit' to stop.")
        print("="*60)
        
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a question.")
                continue
            
            print("\nSearching...")
            result = chatbot.chat_with_routing(query)
            
            print(f"\nAssistant: {result['response']}")
            
            if result['sources']:
                print(f"\nSources ({result['num_sources']} documents):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['title']} (Page {source['page']}) - Score: {source['score']}")
                    if 'url' in source:
                        print(f"     URL: {source['url']}") 
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        logger.error(f"Error in interactive chat: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--process":
            # Process PDFs
            chatbot = LocalRAGChatbot()
            PDF_FOLDER = "./pdfs"
            chatbot.process_pdf_folder(PDF_FOLDER)
        elif sys.argv[1] == "--info":
            # Show collection info
            chatbot = LocalRAGChatbot()
            chatbot.get_collection_info()
    else:
        # Interactive chat
        interactive_chat()