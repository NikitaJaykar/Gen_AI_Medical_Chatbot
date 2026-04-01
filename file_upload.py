from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load File (PDF or TXT)
file_path = "hypertension.pdf"   # change to your file

if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    loader = TextLoader(file_path)

documents = loader.load()

print(f"Loaded {len(documents)} documents")

# 2. Split Documents (Recursive Splitter)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# 3. Load Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create FAISS Vector Store
vector_db = FAISS.from_documents(chunks, embeddings)

# 5. Save FAISS Index Locally
vector_db.save_local("faiss_index")

print("FAISS index saved successfully!")



