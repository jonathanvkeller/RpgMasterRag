# Software Design Document: Tabletop RPG Master RAG System

## Document Information
- **Project Name**: RPG Master RAG POC
- **Version**: 2.0
- **Date**: July 2025
- **Purpose**: Proof of Concept for RAG-based tabletop RPG assistant
- **Approach**: Iterative development with working increments

## 1. Project Overview

### 1.1 Executive Summary
A Retrieval-Augmented Generation (RAG) system that demonstrates advanced prompt engineering and AI integration capabilities through a tabletop RPG rules assistant. This POC prioritizes getting a working RAG pipeline first, then iteratively adding complexity.

### 1.2 Business Objectives
- Demonstrate RAG implementation expertise for job application
- Showcase C# development skills with AI integration
- Prove AI-first development methodology effectiveness
- Create portfolio piece demonstrating semantic search capabilities

### 1.3 Technical Objectives
- Implement complete RAG pipeline with semantic search
- Integrate OpenAI API for embeddings and text generation
- Create maintainable, well-documented codebase
- Achieve sub-2-second query response times
- Build incrementally with working demos at each stage

## 2. System Architecture

### 2.1 High-Level Architecture
```
User Query → Query Processor → Vector Search → Context Retrieval → LLM Generation → Formatted Response
                ↓
            Manual Content → Text Chunking → Embedding Generation → Vector Storage
```

### 2.2 Technology Stack
- **Language**: Python 3.11+
- **Framework**: Python CLI application (with Rich for UI)
- **AI Libraries**: OpenAI Python SDK, LangChain (optional)
- **Vector Database**: Start with in-memory, then ChromaDB or Pinecone
- **Database**: JSON file storage initially, PostgreSQL with pgvector later
- **AI Services**: OpenAI API (text-embedding-3-small, GPT-4o-mini)
- **Key Packages**:
  - `openai` - OpenAI API client
  - `numpy` - Vector operations
  - `pydantic` - Data validation
  - `python-dotenv` - Configuration
  - `rich` - Console UI
  - `chromadb` - Vector storage (Phase 2+)

### 2.3 Implementation Phases

#### Phase 1: Core RAG Pipeline (MVP)
- Manual text chunks (D&D rules as hardcoded strings)
- In-memory vector storage using NumPy arrays
- Basic cosine similarity search with NumPy
- OpenAI integration for embeddings and generation
- Rich console interface for queries

#### Phase 2: Vector Database Integration
- ChromaDB for persistent vector storage
- Embedding caching to avoid re-computation
- Enhanced query interface with history
- JSON config management

#### Phase 3: Database Integration
- PostgreSQL with pgvector extension
- Proper chunk metadata and relationships
- Performance optimization and indexing

#### Phase 4: Advanced Features
- Document ingestion pipeline (PDF, Markdown parsing)
- Advanced retrieval strategies (hybrid search)
- FastAPI web interface or REST API

## 3. Data Models

### 3.1 Phase 1 Models (In-Memory)
```python
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import numpy as np
from numpy.typing import NDArray

class DocumentChunk(BaseModel):
    id: str
    content: str
    source: str
    category: str
    embedding: Optional[NDArray[np.float32]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    similarity_score: float
    citation: str

    class Config:
        arbitrary_types_allowed = True

class QueryResult(BaseModel):
    query: str
    response: str
    source_chunks: List[RetrievedChunk]
    processing_time_ms: float

    class Config:
        arbitrary_types_allowed = True
```

### 3.2 Phase 3 Database Schema (PostgreSQL with pgvector)
```sql
-- PostgreSQL with pgvector extension for native vector operations
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    source VARCHAR(200) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    metadata JSONB, -- Native JSON support
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunk_embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(1536), -- pgvector type for OpenAI embeddings
    embedding_model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for fast similarity search
CREATE INDEX ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops);
```

## 4. Technical Implementation

### 4.1 Phase 1: MVP Implementation

#### 4.1.1 Sample Content (Hardcoded)
```python
def get_sample_chunks() -> List[DocumentChunk]:
    """Returns hardcoded D&D 5e rule chunks for initial testing."""
    return [
        DocumentChunk(
            id="combat-actions",
            content="On your turn, you can move a distance up to your speed and take one action. "
                   "You decide whether to move first or take your action first. Your speed—sometimes "
                   "called your walking speed—is noted on your character sheet.",
            source="Player's Handbook",
            category="Combat",
            metadata={"chapter": "Combat", "page": 189}
        ),
        DocumentChunk(
            id="sneak-attack",
            content="Once per turn, you can deal an extra 1d6 damage to one creature you hit with an "
                   "attack if you have advantage on the attack roll. The attack must use a finesse or "
                   "a ranged weapon. You don't need advantage if another enemy of the target is within "
                   "5 feet of it, that enemy isn't incapacitated, and you don't have disadvantage.",
            source="Player's Handbook",
            category="Class Features",
            metadata={"chapter": "Classes", "class": "Rogue", "page": 96}
        ),
        # More sample chunks...
    ]
```

#### 4.1.2 Vector Storage Service
```python
from abc import ABC, abstractmethod
from typing import List, Protocol
import numpy as np

class EmbeddingService(Protocol):
    """Protocol for embedding generation services."""
    async def generate_embedding(self, text: str) -> NDArray[np.float32]:
        ...

class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""

    @abstractmethod
    async def initialize(self, chunks: List[DocumentChunk]) -> None:
        """Initialize the vector store with document chunks."""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Search for relevant chunks using a text query."""
        pass

    @abstractmethod
    async def search_by_embedding(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Search for relevant chunks using an embedding vector."""
        pass

class InMemoryVectorStore(VectorStore):
    """In-memory vector store using NumPy for similarity search."""

    def __init__(self, embedding_service: EmbeddingService):
        self._chunks: List[DocumentChunk] = []
        self._embedding_service = embedding_service

    async def initialize(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for all chunks that don't have them."""
        self._chunks = chunks

        for chunk in self._chunks:
            if chunk.embedding is None:
                chunk.embedding = await self._embedding_service.generate_embedding(
                    chunk.content
                )

    async def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Search using a text query."""
        query_embedding = await self._embedding_service.generate_embedding(query)
        return await self.search_by_embedding(query_embedding, top_k)

    async def search_by_embedding(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Search using an embedding vector."""
        results = []

        for chunk in self._chunks:
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            results.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=float(similarity),
                citation=f"[{chunk.source} - {chunk.category}]"
            ))

        # Sort by similarity (highest first) and take top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        return dot_product / (magnitude_a * magnitude_b)
```

### 4.2 OpenAI Integration

#### 4.2.1 Configuration
```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class OpenAIConfig(BaseSettings):
    """OpenAI API configuration loaded from environment variables."""
    api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    completion_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout_seconds: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

#### 4.2.2 Services
```python
from openai import AsyncOpenAI
import numpy as np
from typing import List

class OpenAIEmbeddingService:
    """Service for generating embeddings using OpenAI API."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries
        )

    async def generate_embedding(self, text: str) -> NDArray[np.float32]:
        """Generate embedding vector for the given text."""
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

class OpenAIGenerationService:
    """Service for text generation using OpenAI API."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries
        )

    async def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        response = await self.client.chat.completions.create(
            model=self.config.completion_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
```

### 4.3 RAG Pipeline

#### 4.3.1 Query Handler
```python
import time
from typing import List

class RagQueryHandler:
    """Handles RAG queries by retrieving context and generating responses."""

    def __init__(
        self,
        vector_store: VectorStore,
        generation_service: OpenAIGenerationService
    ):
        self.vector_store = vector_store
        self.generation_service = generation_service

    async def handle_query(self, user_query: str) -> QueryResult:
        """Process a user query through the RAG pipeline."""
        start_time = time.time()

        # 1. Retrieve relevant chunks
        retrieved_chunks = await self.vector_store.search(user_query, top_k=3)

        # 2. Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # 3. Generate response using LLM
        prompt = self._build_prompt(user_query, context)
        response = await self.generation_service.generate_response(prompt)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return QueryResult(
            query=user_query,
            response=response,
            source_chunks=retrieved_chunks,
            processing_time_ms=processing_time_ms
        )

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"Source: {chunk.citation}\n"
                f"Content: {chunk.chunk.content}"
            )
        return "\n\n".join(context_parts)

    def _build_prompt(self, user_query: str, context: str) -> str:
        """Build the final prompt for the LLM."""
        return f"""You are an expert D&D 5e rules assistant. Answer the user's question using only the provided context.

Context:
{context}

User Question: {user_query}

Requirements:
- Provide accurate, specific answers based on the context
- Include citations in the format provided
- If information is incomplete, say so
- Use clear, helpful language

Answer:"""
```

## 5. Implementation Plan

### 5.1 Phase 1: Core RAG Pipeline (Day 1)
1. **Project Setup** (1 hour)
   - Create Python virtual environment
   - Install dependencies: `openai`, `numpy`, `pydantic`, `python-dotenv`, `rich`
   - Set up `.env` file for API keys
   - Configure project structure

2. **OpenAI Integration** (2 hours)
   - Implement embedding service with async OpenAI client
   - Implement generation service
   - Add configuration using Pydantic Settings
   - Error handling and retry logic

3. **Vector Storage** (2 hours)
   - Create in-memory vector store with NumPy
   - Implement cosine similarity function
   - Add sample D&D content as Python data

4. **RAG Pipeline** (3 hours)
   - Implement async query handler
   - Create prompt templates
   - Add Rich console interface with formatting
   - Basic async main entry point

### 5.2 Phase 2: Vector Database & Polish (Day 2)
1. **ChromaDB Integration** (2 hours)
   - Install and configure ChromaDB
   - Implement ChromaDB vector store
   - Persistent embedding storage
   - Migration from in-memory

2. **Enhanced Interface** (2 hours)
   - Rich console UI with panels and tables
   - Query history persistence
   - Performance metrics display
   - Interactive query loop

3. **Testing & Documentation** (4 hours)
   - Pytest unit tests for core components
   - Async test fixtures
   - Type checking with mypy
   - README with setup instructions

### 5.3 Phase 3: Database Integration (Future)
- PostgreSQL with pgvector extension
- SQLAlchemy ORM models
- Proper entity relationships
- Query optimization and indexing

## 6. Success Criteria

### 6.1 Phase 1 Success Metrics
- Successfully generate embeddings for sample content
- Retrieve relevant chunks for test queries
- Generate coherent responses with citations
- Response time under 5 seconds for initial implementation

### 6.2 Test Queries
```python
TEST_QUERIES = [
    "How does sneak attack work?",
    "What are the rules for casting spells?",
    "Can you explain initiative in combat?",
    "How do saving throws work?",
    "What's the difference between a bonus action and an action?",
]

async def run_test_queries(query_handler: RagQueryHandler) -> None:
    """Run test queries and display results."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    for query in TEST_QUERIES:
        result = await query_handler.handle_query(query)

        console.print(Panel(
            f"[bold cyan]Query:[/] {result.query}\n\n"
            f"[bold green]Response:[/] {result.response}\n\n"
            f"[bold yellow]Processing Time:[/] {result.processing_time_ms:.2f}ms",
            title="Query Result"
        ))
```

## 7. Project Structure

```
rpg-master-rag/
├── src/
│   └── rpg_master_rag/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── document.py       # Pydantic models
│       ├── services/
│       │   ├── __init__.py
│       │   ├── embedding.py      # OpenAI embedding service
│       │   ├── generation.py     # OpenAI generation service
│       │   └── vector_store.py   # Vector storage implementations
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py       # Pydantic Settings
│       ├── rag/
│       │   ├── __init__.py
│       │   └── query_handler.py  # RAG pipeline
│       ├── data/
│       │   ├── __init__.py
│       │   └── sample_content.py # Hardcoded D&D rules
│       └── main.py               # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_vector_store.py
│   │   └── test_query_handler.py
│   └── integration/
│       └── test_rag_pipeline.py
├── .env.example
├── .gitignore
├── pyproject.toml              # Poetry/pip dependencies
├── requirements.txt            # Alternative to pyproject.toml
├── README.md
└── LICENSE
```

## 8. Key Decisions & Reasoning

### 8.1 Why Start Simple?
- **Reduced Risk**: Each phase delivers working functionality
- **Faster Feedback**: Can test RAG concepts immediately
- **Cleaner Code**: Simpler requirements lead to better architecture
- **Easier Debugging**: Fewer moving parts to troubleshoot

### 8.2 Why In-Memory First?
- **No External Dependencies**: Works on any machine
- **Predictable Performance**: No database setup issues
- **Faster Development**: No schema management overhead
- **Easy Testing**: Deterministic behavior for unit tests

### 8.3 Why Manual Content Initially?
- **Focus on Core Logic**: RAG pipeline is the main challenge
- **Known Good Data**: Eliminates parsing issues
- **Faster Iteration**: No document processing bugs
- **Clear Success Metrics**: Can verify accuracy easily

This revised approach prioritizes working software at each stage while building toward the full vision. Each phase delivers demonstrable value and can be shown to stakeholders.