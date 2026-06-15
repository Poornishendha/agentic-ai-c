import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents(doc_path: str) -> list[Document]:
    """
    Load either:
    1. A single PDF file
    2. All PDFs inside a folder
    """
    docs = []

    path = Path(doc_path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Single PDF file
    if path.is_file():
        print(f"Loading PDF: {path}")

        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())

    # Folder containing PDFs
    else:
        pdf_files = list(path.rglob("*.pdf"))

        print(f"Found {len(pdf_files)} PDF(s)")

        for pdf in pdf_files:
            print(f"Loading PDF: {pdf}")

            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(docs)


def build_vector_store(chunks: list[Document]) -> Chroma:

    if not chunks:
        raise ValueError(
            "No chunks created. Check whether the PDF was loaded correctly."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )


def build_agent(vector_store: Chroma):

    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 2}),
        name="search_document_content",
        description="Search relevant answers from PDF documents.",
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt="You answer questions using the PDF documents.",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        default=str(Path(__file__).parent.parent / "Sample_mom.pdf"),
        help="Path to a PDF file or folder containing PDFs",
    )

    args = parser.parse_args()

    doc_path = str(Path(args.repo).resolve())

    print(f"PDF Path: {doc_path}")
    print(f"Exists: {Path(doc_path).exists()}")

    # Load documents
    docs = load_documents(doc_path)

    if not docs:
        raise ValueError(
            f"No pages were loaded from: {doc_path}"
        )

    # Chunk documents
    chunks = chunk_documents(docs)

    print(
        f"Loaded {len(docs)} pages → "
        f"{len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE})"
    )

    # Build vector store
    vector_store = build_vector_store(chunks)

    # Build agent
    agent = build_agent(vector_store)

    print("\nReady. Ask your question.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            break

        try:
            for step in agent.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                },
                stream_mode="values",
            ):
                last_msg = step["messages"][-1]

                if not getattr(last_msg, "tool_calls", None):
                    print(f"\nAgent: {last_msg.content}")

        except Exception as e:
            print(f"\nError: {e}")