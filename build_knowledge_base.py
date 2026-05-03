import os
import glob
import chromadb
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_PATH = "chroma_db"    # folder where ChromaDB persists to disk
COLLECTION  = "stock_sense"  # ChromaDB collection name
DOCS_FOLDER = "docs_ocr"         # folder containing all your SEC PDFs


# -----------------------------
# STEP 1 — SCAN DOCS FOLDER
# -----------------------------
def find_pdfs(folder: str) -> list[str]:
    """
    Scans the /docs_ocr folder and returns the path of every PDF found.
    Any PDF dropped into /docs_ocr is automatically picked up — no hardcoding needed.
    """
    pdf_paths = glob.glob(os.path.join(folder, "*.pdf"))

    if not pdf_paths:
        print(f"  No PDFs found in '{folder}/' — make sure your files are in place.")
    else:
        print(f"  Found {len(pdf_paths)} PDF(s):")
        for path in pdf_paths:
            print(f"    - {os.path.basename(path)}")

    return pdf_paths


# -----------------------------
# STEP 2 — LOAD ALL PDFS
# -----------------------------
def load_pdfs(pdf_paths: list[str]) -> list[dict]:
    """
    Loads every PDF page by page using LangChain's PyPDFLoader.
    Each page becomes a document dict with title, text, source, and page number.
    The filename (without extension) is used as the document title so ChromaDB
    metadata reflects which SEC guide each chunk came from.
    """
    all_docs = []

    for path in pdf_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(f"\n  Loading: {filename}.pdf")

        try:
            with pdfplumber.open(path) as pdf:
                page_count = 0
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()

                    if not text or len(text.strip()) < 50:
                        continue

                    all_docs.append({
                        "title": filename,
                        "text": text.strip(),
                        "source": path,
                        "page": i + 1
                    })
                    page_count += 1

                print(f"    Loaded {page_count} pages")

        except Exception as e:
            print(f"    Error loading {filename}.pdf: {e}")
            continue

    return all_docs


# -----------------------------
# STEP 3 — CHUNK TEXT
# -----------------------------
def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Splits each page's text into smaller overlapping chunks.
    Smaller chunks = more precise retrieval when the agent queries the DB.
    Each chunk inherits the parent document's metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # ~500 characters per chunk — good balance for definitions
        chunk_overlap=50  # 50 char overlap so context isn't cut off at boundaries
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])

        for i, split in enumerate(splits):
            # Build a unique ID from source path + page + chunk index
            # This ensures upsert never creates duplicates on re-runs
            chunk_id = f"{doc['source']}__page{doc['page']}__chunk{i}"

            chunks.append({
                "id":     chunk_id,
                "text":   split,
                "title":  doc["title"],
                "source": doc["source"],
                "page":   doc["page"]
            })

    return chunks


# -----------------------------
# STEP 4 — EMBED + STORE IN CHROMADB
# -----------------------------
def store_in_chroma(chunks: list[dict]):
    """
    Embeds all text chunks using OpenAI text-embedding-3-small and stores
    them in a persistent ChromaDB collection on disk.
    Uses upsert so re-running the script never creates duplicate entries.
    Processes in batches of 50 to stay within OpenAI rate limits.
    """
    print(f"\n  Embedding and storing {len(chunks)} chunks...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # PersistentClient saves the DB to disk at CHROMA_PATH
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION)

    batch_size    = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch     = chunks[i:i + batch_size]
        texts     = [c["text"]   for c in batch]
        ids       = [c["id"]     for c in batch]
        metadatas = [
            {
                "title":  c["title"],
                "source": c["source"],
                "page":   str(c["page"])  # ChromaDB requires metadata values as strings
            }
            for c in batch
        ]

        vectors = embeddings.embed_documents(texts)

        collection.upsert(
            ids        = ids,
            embeddings = vectors,
            documents  = texts,
            metadatas  = metadatas
        )

        print(f"    Batch {i // batch_size + 1}/{total_batches} stored")

    print(f"\n  Done. {len(chunks)} chunks saved to '{CHROMA_PATH}/'")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def build_knowledge_base():
    print("=" * 60)
    print("  StockSense — RAG Knowledge Base Builder")
    print("=" * 60)

    # --- Step 1: Find all PDFs ---
    print(f"\n[1/4] Scanning '{DOCS_FOLDER}/' for PDFs...")
    pdf_paths = find_pdfs(DOCS_FOLDER)

    if not pdf_paths:
        print("\nNo PDFs to process. Exiting.")
        return

    # --- Step 2: Load all PDFs ---
    print(f"\n[2/4] Loading PDF content...")
    documents = load_pdfs(pdf_paths)
    print(f"\n  Total pages loaded: {len(documents)}")

    if not documents:
        print("\nNo content extracted from PDFs. Check your files and try again.")
        return

    # --- Step 3: Chunk ---
    print(f"\n[3/4] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Total chunks created: {len(chunks)}")

    # --- Step 4: Embed + Store ---
    print(f"\n[4/4] Embedding and storing in ChromaDB...")
    store_in_chroma(chunks)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Knowledge base build complete!")
    print(f"  PDFs processed : {len(pdf_paths)}")
    print(f"  Pages loaded   : {len(documents)}")
    print(f"  Chunks stored  : {len(chunks)}")
    print(f"  ChromaDB path  : {CHROMA_PATH}/")
    print("=" * 60)
    print("\nYou can now use query_knowledge_base() in your agent.")


if __name__ == "__main__":
    build_knowledge_base()