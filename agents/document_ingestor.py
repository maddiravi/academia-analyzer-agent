import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentIngestorAgent:
    """Agent 1: Handles local document loading, splitting, and RAG index creation."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, # Optimized for academic text density
            chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""]
        )
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"DocumentIngestorAgent: File not found at {self.file_path}")

    def _load_document(self):
        """Loads content based on file extension (PDF or TXT/MD)."""
        ext = os.path.splitext(self.file_path)[1].lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(self.file_path)
        elif ext in ['.txt', '.md']:
            loader = TextLoader(self.file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .pdf, .txt, or .md supported.")
        
        return loader.load()

    def process_document(self):
        """Loads, splits, and processes the document."""
        print(f"Loading document: {self.file_path}...")
        docs = self._load_document()
        
        chunks = self.text_splitter.split_documents(docs)
        print(f"Split content into {len(chunks)} chunks.")
        
        full_content = "\n".join([c.page_content for c in chunks])
        
        return chunks, full_content

    def create_retriever(self, chunks):
        """Creates and returns a FAISS-based retriever from the document chunks."""
        print("Creating RAG retriever...")
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        retriever = vectorstore.as_retriever()
        return retriever