
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
import chromadb
import tempfile
import os
import pickle

def create_vector_store_for_hf_spaces():
    """
    Create vector store compatible with Hugging Face Spaces
    Uses EphemeralClient and loads data dynamically
    """
    print("Loading dataset for Hugging Face Spaces...")
    
    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Use EphemeralClient for Hugging Face Spaces
    chroma_client = chromadb.EphemeralClient()
    
    db = Chroma(
        client=chroma_client,
        collection_name="example_collection",
        embedding_function=huggingface_embeddings
    )

    # Try to load from MentalChat16K dataset
    try:
        print("Attempting to load MentalChat16K dataset...")
        ds = load_dataset("ShenLab/MentalChat16K", split="train[:100]")  # Load only first 100 for speed
        
        docs = []
        for row in ds:
            user_input = row['input']
            bot_reply = row['output']
            docs.append(Document(
                page_content=f"Q: {user_input}\nA: {bot_reply}",
                metadata={"source": "MentalChat16K"}
            ))
        
        print(f"Loaded {len(docs)} documents from dataset.")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using fallback sample data...")
        
        # Fallback sample data
        docs = [
            Document(page_content="Q: I'm feeling anxious about work\nA: It's completely normal to feel anxious about work. Try taking deep breaths and breaking down your tasks into smaller, manageable steps.", metadata={"source": "sample"}),
            Document(page_content="Q: I can't sleep because I'm worried\nA: Worry can definitely affect sleep. Try creating a calming bedtime routine and writing down your worries in a journal to get them out of your mind.", metadata={"source": "sample"}),
            Document(page_content="Q: I feel overwhelmed with everything\nA: Feeling overwhelmed is a sign that you're taking on a lot. It's okay to ask for help and prioritize what's most important right now.", metadata={"source": "sample"}),
            Document(page_content="Q: I'm having relationship problems\nA: Relationships can be challenging. Open and honest communication is key. Consider talking to your partner about how you're feeling.", metadata={"source": "sample"}),
            Document(page_content="Q: I feel sad and don't know why\nA: Sometimes sadness comes without a clear reason, and that's okay. Be gentle with yourself and consider talking to someone you trust about how you're feeling.", metadata={"source": "sample"}),
            Document(page_content="Q: I'm stressed about my future\nA: It's natural to feel uncertain about the future. Focus on what you can control today and take small steps toward your goals.", metadata={"source": "sample"}),
            Document(page_content="Q: I feel lonely and isolated\nA: Loneliness is a difficult emotion. Consider reaching out to a friend, joining a community group, or even talking to a counselor who can provide support.", metadata={"source": "sample"}),
            Document(page_content="Q: I'm having panic attacks\nA: Panic attacks can be frightening, but they are treatable. Try grounding techniques like deep breathing and consider speaking with a mental health professional.", metadata={"source": "sample"}),
            Document(page_content="Q: I don't feel motivated to do anything\nA: Loss of motivation can be a sign of depression or burnout. Be patient with yourself and consider starting with very small, achievable tasks.", metadata={"source": "sample"}),
            Document(page_content="Q: I'm dealing with grief\nA: Grief is a natural process that takes time. Allow yourself to feel your emotions and consider joining a support group or talking to a counselor.", metadata={"source": "sample"})
        ]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    chunked_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(chunked_docs)} chunks.")

    # Add documents to the vector store
    db.add_documents(chunked_docs)
    print("✅ Vector store created successfully for Hugging Face Spaces.")

    return db
