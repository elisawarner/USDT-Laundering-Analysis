from typing import TypedDict, List, Optional
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

def detect_chapter(text: str, page_num: int) -> Optional[str]:
    """
    Detect chapter title from text using common patterns.

    Args:
        text: Text to search for chapter markers
        page_num: Current page number

    Returns:
        Chapter title if found, None otherwise
    """
    # Common chapter patterns
    patterns = [
        r'^Chapter\s+(\d+|[IVXLCDM]+)[:\s]+(.+?)$',  # Chapter 1: Title or Chapter I: Title
        r'^CHAPTER\s+(\d+|[IVXLCDM]+)[:\s]+(.+?)$',  # CHAPTER 1: Title
        r'^(\d+)\.\s+(.+?)$',                        # 1. Title
        r'^Part\s+(\d+|[IVXLCDM]+)[:\s]+(.+?)$',     # Part 1: Title
        r'^Section\s+(\d+)[:\s]+(.+?)$'              # Section 1: Title
    ]

    lines = text.split('\n')
    for i, line in enumerate(lines[:5]):  # Check first 5 lines of page
        line = line.strip()
        if not line:
            continue

        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    chapter_num, chapter_title = match.groups()[0], match.groups()[1]
                    return f"Chapter {chapter_num}: {chapter_title.strip()}"
                else:
                    return line.strip()

    return None

    
def split_into_paragraphs(text:str) -> List[str]:
    """
    Split text into paragraphs.

    Args:
        text: Text to split

    Returns:
        List of paragraphs
    """
    # Split by double newlines (standard paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)

    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs

    
def load_pdf_with_citations(pdf_path: str) -> List[Document]:
    """
    Load a PDF and extract text with chapter and paragraph metadata.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects with detailed citation metadata
    """
    documents = []
    current_chapter = "Introduction"  # Default chapter

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        print(f"Loading {total_pages} pages from {os.path.basename(pdf_path)}...")

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()

            if not text.strip():
                continue

            # Check if this page starts a new chapter
            detected_chapter = detect_chapter(text, page_num + 1)
            if detected_chapter:
                current_chapter = detected_chapter
                print(f"  📖 Found: {current_chapter} (Page {page_num + 1})")

            # Split page into paragraphs
            paragraphs = split_into_paragraphs(text)

            # Create a document for each paragraph
            for para_num, paragraph in enumerate(paragraphs, 1):
                if len(paragraph) < 50:  # Skip very short paragraphs (likely headers/footers)
                    continue

                doc = Document(
                    page_content=paragraph,
                    metadata={
                        "page": page_num + 1,
                        "chapter": current_chapter,
                        "paragraph": para_num,
                        "source": pdf_path
                    }
                )
                documents.append(doc)

    print(f"✅ Loaded {len(documents)} paragraphs from {total_pages} pages")
    return documents

    
def chunk_documents_with_metadata(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks while preserving citation metadata.

    Args:
        documents: List of Document objects

    Returns:
        List of chunks Document objects with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks with citation metadata")

    return chunks

    
def format_citation(page: int, chapter: str, paragraph: int) -> str:
    """
    Format a citation in academic style.

    Args:
        page: Page number
        chapter: Chapter title
        paragraph: Paragraph number

    Returns:
        Formatted citation string
    """
    return f"({chapter}, Page {page}, ¶{paragraph})"
    
print("✅ Helper functions with citation support defined!")