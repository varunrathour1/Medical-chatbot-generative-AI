from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Extract Data From the PDF File
def load_pdf_file(data_path):
    """
    Load PDF files from the specified directory.
    Args:
        data_path (str): Path to the directory containing PDF files.
    Returns:
        List of documents.
    """
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Split the Data into Text Chunks
def text_split(extracted_data):
    """
    Split extracted data into smaller chunks for processing.
    Args:
        extracted_data: List of documents.
    Returns:
        List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    """
    Download HuggingFace embeddings model.
    Returns:
        HuggingFaceEmbeddings instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'  # Embedding size is 384 dimensions
    )
    return embeddings
