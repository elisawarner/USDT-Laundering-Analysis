from typing import TypedDict, List, Optional
from langchain_core.tools import tool, InjectedToolArg
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import PyPDF2
import os
import warnings
import re
from book_helper import detect_chapter, split_into_paragraphs, load_pdf_with_citations, chunk_documents_with_metadata, format_citation

warnings.filterwarnings('ignore')


class BookSearchState(TypedDict):
    """State for the book search agent"""

    # Input
    query: str                       # User's search query
    book_path: str                   # Path to the PDF book

    # Intermediate data
    vectorstore: Optional[any]       # FAISS 
    documents: List[Document]        # Processed book chunks
    relevant_chunks: List[dict]      # Retrieved relevant sections

    # Output
    answer: str                      # Generated answer
    locations: List[dict]            # Page numbers and snippets
    citations: List[str]             # Formatted citations

    # Metadata
    search_strategy: str             # "semantic" or "keyword"
    retry_count: int                 # Number of serach attempts

def load_book_node(state: BookSearchState) -> BookSearchState:
    """
    Load book and create vector store for semantic search with citation metadata.
    Caches embeddings to avoid re-processing.
    """
    book_path = state["book_path"]
    cache_path = f"{book_path}.faiss"

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Check if we have cached embeddings
    if os.path.exists(cache_path):
        print("📂 Loading cached embeddings...")
        vectorstore = FAISS.load_local(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        state["vectorstore"] = vectorstore
        print("✅ Loaded from cache!")
    else:
        print("📖 Processing book with citation tracking...")

        # Load PDF with chapter and paragraph metadata
        documents = load_pdf_with_citations(book_path)

        # Chunk documents while preserving metadata
        chunks = chunk_documents_with_metadata(documents)

        # Create vector store (this creates embeddings via API)
        print("🔄 Creating embeddings (this may take a minute)...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save to cache
        vectorstore.save_local(cache_path)
        print(f"💾 Embeddings cached to {cache_path}")

        state["vectorstore"] = vectorstore
        state["documents"] = chunks

    return state

def analyze_query_node(state: BookSearchState) -> BookSearchState:
    """
    Analyze the user's query to determine search strategy.
    For now, we'll default ot semantic search.
    """
    query = state["query"]

    # Simple heuristic: use semantic search for conceptual queries
    # Use keyword search for exact phrase matches (in quotes)
    if '"' in query:
        state["search_strategy"] = "keyword"
        print("🔍 Using keyword search strategy")
    else:
        state["search_strategy"] = "semantic"
        print("🧠 Using semantic search strategy")

    # Initialize retry count
    if "retry_count" not in state:
        state["retry_count"] = 0

    return state

def search_content_node(state: BookSearchState) -> BookSearchState:
    """
    Search for relevant content using semantic similarity.
    The vector store handles embedding comparison automatically.
    Preserves citation metadata for each result.
    """
    query = state["query"]
    vectorstore = state["vectorstore"]

    print(f"🔍 Searching for: '{query}'")

    # Perform similarity search
    # This automatically embeds the query and compares to stored embeddings
    results = vectorstore.similarity_search_with_score(
        query,
        k=5  # Return top 5 most relevant chunks
    )

    # Format results with citation metadata
    relevant_chunks = []
    for doc, score in results:
        relevant_chunks.append({
            "text": doc.page_content,
            "page": doc.metadata.get("page", "Unknown"),
            "chapter": doc.metadata.get("chapter", "Unknown Chapter"),
            "paragraph": doc.metadata.get("paragraph", 1),
            "source": doc.metadata.get("source", ""),
            "similarity_score": float(score)
        })

    state["relevant_chunks"] = relevant_chunks

    print(f"✅ Found {len(relevant_chunks)} relevant chunks with citations")
    for i, chunk in enumerate(relevant_chunks[:3]):
        citation = format_citation(chunk['page'], chunk['chapter'], chunk['paragraph'])
        print(f"  {i+1}. {citation} (score: {chunk['similarity_score']:.3f})")

    return state

def extract_locations_node(state: BookSearchState) -> BookSearchState:
    """
    Extract and format locations with full citation information.
    """
    relevant_chunks = state["relevant_chunks"]

    locations = []
    citations = []

    for chunk in relevant_chunks:
        # Create a snippet (first 200 characters)
        snippet = chunk["text"][:200]
        if len(chunk["text"]) > 200:
            snippet += "..."

        # Format citation
        citation = format_citation(
            chunk["page"],
            chunk["chapter"],
            chunk["paragraph"]
        )

        locations.append({
            "page": chunk["page"],
            "chapter": chunk["chapter"],
            "paragraph": chunk["paragraph"],
            "snippet": snippet,
            "full_text": chunk["text"],
            "citation": citation,
            "relevance": chunk["similarity_score"]
        })

        citations.append(citation)

    state["locations"] = locations
    state["citations"] = citations

    print(f"📍 Extracted {len(locations)} locations with citations")

    return state

def generate_answer_node(state: BookSearchState) -> BookSearchState:
    """
    Generate a natural language answer with proper citations.
    """
    query = state["query"]
    relevant_chunks = state["relevant_chunks"]

    if not relevant_chunks:
        state["answer"] = "I couldn't find any relevant information about that in the book."
        return state

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL')
    )

    # Construct context with ctiations
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        citation = format_citation(chunk['page'], chunk['chapter'], chunk['paragraph'])
        context_parts.append(f"[Source {i}] {citation}:\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    # Create prompt with citation instructions
    prompt = f"""Based on the following excerpts from a book, answer the question.
You MUST cite your sources using the [Source X] numbers provided.

Include citations in your answer like this:
- "The author discusses X [Source 1]."
- "According to the text, Y and Z are important [Source 2, Source 3]."

Context from the book:
{context}

Question: {query}

Answer (include [Source X] citations in your reponse):"""

    print("💬 Generating answer with citations...")

    # Generate answer
    response = llm.invoke(prompt)
    answer_text = response.content

    # Add a references section at the end
    references = "\n\n📚 REFERENCES:\n"
    for i, chunk in enumerate(relevant_chunks, 1):
        citation = format_citation(chunk['page'], chunk['chapter'], chunk['paragraph'])
        references += f"[Source {i}] {citation}\n"

    state["answer"] = answer_text + references

    print("✅ Answer with citations generated!")

    return state

@tool
def book_lookup(query: str) -> str:
    """
    Make the other agent a tool
    """
    ###### GRAPH SETUP #######
    # Create the graph
    workflow = StateGraph(BookSearchState)
    
    workflow.add_node("load_book", load_book_node)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_content", search_content_node)
    workflow.add_node("extract_locations", extract_locations_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Define the flow
    workflow.set_entry_point("load_book","analyze_query")
    workflow.add_edge("analyze_query", "search_content")
    
    # Conditional edge: retry search or continue
    def should_retry(state: BookSearchState) -> str:
        """Decide whether to retry search or continue to answer generation."""
        if len(state.get("relevant_chunks", [])) == 0 and state.get("retry_count", 0) < 1:
            state["retry_count"] += 1
            print("🔄 No results found, retrying with different strategy...")
            return "analyze_query"  # Retry
        return "extract_locations"  #Continue
    
    workflow.add_conditional_edges(
        "search_content",
        should_retry,
        {
            "analyze_query": "analyze_query",
            "extract_locations": "extract_locations"
        }
    )
    
    workflow.add_edge("extract_locations", "generate_answer")
    workflow.add_edge("generate_answer", END)
    #####################
    
    # Compile the graph
    app = workflow.compile()
    
    # Configure your search
    BOOK_PATH = "./AML1.pdf"  # Change this to your PDF path
    
    # Create initial state
    initial_state = {
        "query": query,
        "book_path": BOOK_PATH,
        "vectorstore": None,
        "documents": [],
        "relevant_chunks": [],
        "answer": "",
        "locations": [],
        "citations": [],
        "search_strategy": "",
        "retry_count": 0
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    print(result["answer"])
    
    for i, location in enumerate(result["locations"], 1):
        print(f"\n{i}. {location['citation']}")
        print(f"   Relevance Score: {location['relevance']:.3f}")
        print(f"   Snippet: {location['snippet']}")
        print()
    return result