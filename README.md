# RPG Master RAG

A Retrieval-Augmented Generation (RAG) system for tabletop RPG rules assistance, built with Python and OpenAI.

## Overview

This project demonstrates a complete RAG pipeline implementation using:
- **Python 3.11+** for modern async/await patterns
- **OpenAI API** for embeddings (text-embedding-3-small) and generation (GPT-4o-mini)
- **NumPy** for vector operations and cosine similarity
- **Pydantic** for data validation and settings management
- **Rich** for beautiful console UI

## Features

- 🔍 Semantic search over tabletop RPG rules using vector embeddings
- 🤖 AI-powered responses with source citations
- ⚡ Async/await for efficient API calls
- 📊 Performance metrics and query tracking
- 🎨 Beautiful console interface with Rich

## Project Status

Currently in **planning phase**. See [revised_sdd.md](./revised_sdd.md) for the complete Software Design Document with architecture details, data models, and implementation plans.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- OpenAI API key

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai numpy pydantic python-dotenv rich

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Quick Start

```python
# Coming soon! Implementation follows the design in revised_sdd.md
```

## Development Phases

1. **Phase 1 (MVP)**: In-memory vector storage with hardcoded D&D rules
2. **Phase 2**: ChromaDB integration for persistent storage
3. **Phase 3**: PostgreSQL with pgvector extension
4. **Phase 4**: Document ingestion pipeline and web API

## Documentation

- [Software Design Document](./revised_sdd.md) - Complete technical specification

## License

See [LICENSE](./LICENSE) file for details.