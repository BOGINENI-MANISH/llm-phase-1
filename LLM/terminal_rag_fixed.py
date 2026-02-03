import os
import sys
import argparse
import logging

# Suppress non-critical warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Prefer community loaders (installed via requirements)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Optional imports - chains may not be present in all langchain builds
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    HAS_CHAINS = True
except Exception:
    create_stuff_documents_chain = None
    create_retrieval_chain = None
    HAS_CHAINS = False

# --- DEFAULT CONFIGURATION ---
DEFAULT_DATA_PATH = "./documents"
DEFAULT_DB_FAISS_PATH = "./vectorstore"
DEFAULT_MODEL_NAME = "llama3"  # Ensure you ran 'ollama pull llama3' if using Ollama


def parse_args():
    parser = argparse.ArgumentParser(description="Simple terminal RAG demo")
    parser.add_argument("--pdf-path", default=DEFAULT_DATA_PATH, help="Path to PDFs directory")
    parser.add_argument("--db-path", default=DEFAULT_DB_FAISS_PATH, help="Directory to store FAISS DB")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="LLM model name (for Ollama)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def retrieve_docs(retriever, query, verbose=False):
    # Handle different retriever API shapes across langchain versions
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "_get_relevant_documents"):
        # some versions expect a keyword-only run_manager argument
        try:
            return retriever._get_relevant_documents(query, run_manager=None)
        except TypeError:
            return retriever._get_relevant_documents(query)
    # As a last resort try invoking the retriever and extracting docs
    if hasattr(retriever, "invoke"):
        try:
            res = retriever.invoke({"input": query})
            if isinstance(res, dict):
                for key in ("data", "documents", "docs", "source_documents"):
                    if key in res:
                        return res[key]
        except Exception as e:
            logging.debug("Retriever.invoke failed: %s", e)
    return []


def call_llm(llm, prompt_text):
    """Try several common invocation patterns for different LLM wrappers.

    Returns the raw result from the LLM call.
    """
    # 1) Try invoke with string
    if hasattr(llm, "invoke"):
        try:
            return llm.invoke(prompt_text)
        except TypeError:
            # Some wrappers expect a dict input
            try:
                return llm.invoke({"input": prompt_text})
            except Exception:
                pass
    # 2) Try calling the llm as a callable
    if callable(llm):
        try:
            return llm(prompt_text)
        except TypeError:
            try:
                return llm({"input": prompt_text})
            except Exception:
                pass
    raise RuntimeError("No compatible invocation method for the provided LLM wrapper")


def fallback_retrieval_and_call(retriever, llm, query, verbose=False):
    # Basic fallback when langchain.chains APIs are not available
    if verbose:
        logging.info("Using fallback retrieval path (manual retrieval + LLM call)")

    docs = retrieve_docs(retriever, query, verbose=verbose)
    if not docs:
        print("❌ No documents returned by retriever.")
        return

    # Concatenate a reasonable amount of context
    context = "\n\n".join(d.page_content for d in docs)
    prompt_text = f"You are an intelligent assistant.\nRead the following context from a document and provide a detailed summary and description of what this document is about.\n\n<context>\n{context}\n</context>\n\nQuestion: {query}"

    try:
        result = call_llm(llm, prompt_text)

        # Normalize and print common result shapes
        if isinstance(result, dict):
            for key in ("output", "answer", "text", "content", "result"):
                if key in result:
                    print(result[key])
                    return
            # Handle 'generations' shape (LangChain LLMResult-like)
            if "generations" in result:
                gens = result["generations"]
                try:
                    if isinstance(gens, list) and gens and isinstance(gens[0], list) and hasattr(gens[0][0], "text"):
                        print(gens[0][0].text)
                        return
                except Exception:
                    pass
            # Fallback to raw repr
            print(result)
        elif isinstance(result, str):
            print(result)
        else:
            # Try to extract typical LangChain LLMResult.generations
            gens = getattr(result, "generations", None)
            if gens:
                try:
                    if isinstance(gens, list) and gens and isinstance(gens[0], list) and hasattr(gens[0][0], "text"):
                        print(gens[0][0].text)
                        return
                except Exception:
                    pass
            # Last resort
            print(result)
    except Exception as e:
        logging.error("LLM invocation failed: %s", e)
        print("❌ LLM invocation failed. Ensure Ollama is running and the model is available.")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    DATA_PATH = args.pdf_path
    DB_FAISS_PATH = args.db_path
    MODEL_NAME = args.model

    # 1. Check if document exists
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"❌ Error: Please put a PDF file inside the '{DATA_PATH}' folder.")
        return

    logging.info("Processing PDFs from: %s", DATA_PATH)

    # 2. Ingest Data (Load & Split)
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("❌ No PDF found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Vector Store
    logging.info("Creating embeddings and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    # 4. Setup Llama 3 Connection
    logging.info("Connecting to Llama (model=%s)...", MODEL_NAME)
    try:
        llm = ChatOllama(model=MODEL_NAME)
    except Exception as e:
        logging.error("Failed to initialize ChatOllama: %s", e)
        llm = None

    # Load the DB we just created
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 5. Define the description prompt (kept simple)
    base_prompt = ChatPromptTemplate.from_template(
        "You are an intelligent assistant.\nRead the following context from a document and provide a detailed summary and description of what this document is about.\n\n<context>\n{context}\n</context>\n\nQuestion: {input}"
    )

    logging.info("Generating document description...")

    query = "Describe this document and summarize its key points."

    # 6. Use the langchain chains API if available, otherwise fallback
    if HAS_CHAINS and create_stuff_documents_chain and create_retrieval_chain and llm is not None:
        try:
            document_chain = create_stuff_documents_chain(llm, base_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": query})
            print(response.get("answer") if isinstance(response, dict) else response)
        except Exception as e:
            logging.warning("Chains API failed: %s. Falling back to manual retrieval.", e)
            if llm is None:
                print("❌ LLM is not available. Skipping generation.")
            else:
                fallback_retrieval_and_call(retriever, llm, query, verbose=args.verbose)
    else:
        if not HAS_CHAINS:
            logging.info("langchain 'chains' APIs not found; using fallback flow.")
        if llm is None:
            print("❌ LLM is not available. Skipping generation.")
        else:
            fallback_retrieval_and_call(retriever, llm, query, verbose=args.verbose)


if __name__ == "__main__":
    main()
