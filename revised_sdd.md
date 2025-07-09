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
- **Framework**: .NET 8 Console Application
- **Database**: Start with JSON file storage, migrate to SQL Server later
- **AI Services**: OpenAI API (text-embedding-3-small, GPT-4o-mini)
- **Document Processing**: Manual text input initially, automated later
- **Vector Operations**: In-memory cosine similarity, then SQL Server

### 2.3 Implementation Phases

#### Phase 1: Core RAG Pipeline (MVP)
- Manual text chunks (D&D rules as hardcoded strings)
- In-memory vector storage using simple arrays
- Basic cosine similarity search
- OpenAI integration for embeddings and generation
- Console interface for queries

#### Phase 2: Persistent Storage
- JSON file-based chunk storage
- Embedding caching to avoid re-computation
- Better query interface

#### Phase 3: Database Integration
- SQL Server with traditional tables
- Proper chunk metadata and relationships
- Performance optimization

#### Phase 4: Advanced Features
- Document ingestion pipeline
- SQL Server vector extensions (if available)
- Web interface or API

## 3. Data Models

### 3.1 Phase 1 Models (In-Memory)
```csharp
public class DocumentChunk
{
    public string Id { get; set; }
    public string Content { get; set; }
    public string Source { get; set; }
    public string Category { get; set; }
    public float[] Embedding { get; set; }
    public Dictionary<string, object> Metadata { get; set; }
}

public class QueryResult
{
    public string Query { get; set; }
    public string Response { get; set; }
    public List<RetrievedChunk> SourceChunks { get; set; }
    public double ProcessingTimeMs { get; set; }
}

public class RetrievedChunk
{
    public DocumentChunk Chunk { get; set; }
    public double SimilarityScore { get; set; }
    public string Citation { get; set; }
}
```

### 3.2 Phase 3 Database Schema
```sql
-- Traditional SQL Server tables (no vector extensions needed initially)
CREATE TABLE Documents (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    Title NVARCHAR(500) NOT NULL,
    Source NVARCHAR(200) NOT NULL,
    ContentType NVARCHAR(100) NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETUTCDATE()
);

CREATE TABLE DocumentChunks (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    DocumentId UNIQUEIDENTIFIER FOREIGN KEY REFERENCES Documents(Id),
    ChunkIndex INT NOT NULL,
    Content NVARCHAR(MAX) NOT NULL,
    Source NVARCHAR(200) NOT NULL,
    Category NVARCHAR(100) NOT NULL,
    Metadata NVARCHAR(MAX) NULL, -- JSON metadata
    CreatedAt DATETIME2 DEFAULT GETUTCDATE()
);

CREATE TABLE ChunkEmbeddings (
    ChunkId UNIQUEIDENTIFIER FOREIGN KEY REFERENCES DocumentChunks(Id),
    EmbeddingData VARBINARY(MAX) NOT NULL, -- Serialized float array
    EmbeddingModel NVARCHAR(100) NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETUTCDATE()
);
```

## 4. Technical Implementation

### 4.1 Phase 1: MVP Implementation

#### 4.1.1 Sample Content (Hardcoded)
```csharp
public static class SampleContent
{
    public static List<DocumentChunk> GetSampleChunks()
    {
        return new List<DocumentChunk>
        {
            new DocumentChunk
            {
                Id = "combat-actions",
                Content = "On your turn, you can move a distance up to your speed and take one action...",
                Source = "Player's Handbook",
                Category = "Combat",
                Metadata = new Dictionary<string, object> { {"chapter", "Combat"}, {"page", 189} }
            },
            // More sample chunks...
        };
    }
}
```

#### 4.1.2 Vector Storage Service
```csharp
public interface IVectorStore
{
    Task InitializeAsync(List<DocumentChunk> chunks);
    Task<List<RetrievedChunk>> SearchAsync(string query, int topK = 5);
    Task<List<RetrievedChunk>> SearchByEmbeddingAsync(float[] queryEmbedding, int topK = 5);
}

public class InMemoryVectorStore : IVectorStore
{
    private List<DocumentChunk> _chunks;
    private readonly IEmbeddingService _embeddingService;
    
    public async Task InitializeAsync(List<DocumentChunk> chunks)
    {
        _chunks = chunks;
        
        // Generate embeddings for all chunks
        foreach (var chunk in _chunks.Where(c => c.Embedding == null))
        {
            chunk.Embedding = await _embeddingService.GenerateEmbeddingAsync(chunk.Content);
        }
    }
    
    public async Task<List<RetrievedChunk>> SearchAsync(string query, int topK = 5)
    {
        var queryEmbedding = await _embeddingService.GenerateEmbeddingAsync(query);
        return await SearchByEmbeddingAsync(queryEmbedding, topK);
    }
    
    public Task<List<RetrievedChunk>> SearchByEmbeddingAsync(float[] queryEmbedding, int topK = 5)
    {
        var results = _chunks
            .Select(chunk => new RetrievedChunk
            {
                Chunk = chunk,
                SimilarityScore = CosineSimilarity(queryEmbedding, chunk.Embedding),
                Citation = $"[{chunk.Source} - {chunk.Category}]"
            })
            .OrderByDescending(r => r.SimilarityScore)
            .Take(topK)
            .ToList();
            
        return Task.FromResult(results);
    }
    
    private static double CosineSimilarity(float[] a, float[] b)
    {
        // Simple cosine similarity implementation
        var dotProduct = a.Zip(b, (x, y) => x * y).Sum();
        var magnitudeA = Math.Sqrt(a.Sum(x => x * x));
        var magnitudeB = Math.Sqrt(b.Sum(x => x * x));
        return dotProduct / (magnitudeA * magnitudeB);
    }
}
```

### 4.2 OpenAI Integration

#### 4.2.1 Configuration
```csharp
public class OpenAIConfig
{
    public string ApiKey { get; set; }
    public string EmbeddingModel { get; set; } = "text-embedding-3-small";
    public string CompletionModel { get; set; } = "gpt-4o-mini";
    public int MaxRetries { get; set; } = 3;
    public int TimeoutSeconds { get; set; } = 30;
}
```

#### 4.2.2 Services
```csharp
public interface IEmbeddingService
{
    Task<float[]> GenerateEmbeddingAsync(string text);
}

public interface IGenerationService
{
    Task<string> GenerateResponseAsync(string prompt);
}
```

### 4.3 RAG Pipeline

#### 4.3.1 Query Handler
```csharp
public class RagQueryHandler
{
    private readonly IVectorStore _vectorStore;
    private readonly IGenerationService _generationService;
    
    public async Task<QueryResult> HandleQueryAsync(string userQuery)
    {
        var stopwatch = Stopwatch.StartNew();
        
        // 1. Retrieve relevant chunks
        var retrievedChunks = await _vectorStore.SearchAsync(userQuery, topK: 3);
        
        // 2. Build context
        var context = BuildContext(retrievedChunks);
        
        // 3. Generate response
        var prompt = BuildPrompt(userQuery, context);
        var response = await _generationService.GenerateResponseAsync(prompt);
        
        stopwatch.Stop();
        
        return new QueryResult
        {
            Query = userQuery,
            Response = response,
            SourceChunks = retrievedChunks,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }
    
    private string BuildContext(List<RetrievedChunk> chunks)
    {
        return string.Join("\n\n", chunks.Select(c => 
            $"Source: {c.Citation}\nContent: {c.Chunk.Content}"));
    }
    
    private string BuildPrompt(string userQuery, string context)
    {
        return $@"You are an expert D&D 5e rules assistant. Answer the user's question using only the provided context.

Context:
{context}

User Question: {userQuery}

Requirements:
- Provide accurate, specific answers based on the context
- Include citations in the format provided
- If information is incomplete, say so
- Use clear, helpful language

Answer:";
    }
}
```

## 5. Implementation Plan

### 5.1 Phase 1: Core RAG Pipeline (Day 1)
1. **Project Setup** (1 hour)
   - Create .NET 8 console application
   - Add OpenAI NuGet packages
   - Configure dependency injection

2. **OpenAI Integration** (2 hours)
   - Implement embedding service
   - Implement generation service
   - Add configuration and error handling

3. **Vector Storage** (2 hours)
   - Create in-memory vector store
   - Implement cosine similarity
   - Add sample D&D content

4. **RAG Pipeline** (3 hours)
   - Implement query handler
   - Create prompt templates
   - Add basic console interface

### 5.2 Phase 2: Persistence & Polish (Day 2)
1. **File-based Storage** (2 hours)
   - JSON serialization for chunks
   - Embedding caching
   - Configuration management

2. **Enhanced Interface** (2 hours)
   - Better console UI
   - Query history
   - Performance metrics display

3. **Testing & Documentation** (4 hours)
   - Unit tests for core components
   - Integration tests
   - README and documentation

### 5.3 Phase 3: Database Integration (Future)
- SQL Server integration
- Proper entity relationships
- Performance optimization

## 6. Success Criteria

### 6.1 Phase 1 Success Metrics
- Successfully generate embeddings for sample content
- Retrieve relevant chunks for test queries
- Generate coherent responses with citations
- Response time under 5 seconds for initial implementation

### 6.2 Test Queries
```csharp
public static class TestQueries
{
    public static List<string> GetTestQueries()
    {
        return new List<string>
        {
            "How does sneak attack work?",
            "What are the rules for casting spells?",
            "Can you explain initiative in combat?",
            "How do saving throws work?",
            "What's the difference between a bonus action and an action?"
        };
    }
}
```

## 7. Project Structure

```
RpgMasterRAG/
├── src/
│   ├── RpgMaster.Core/
│   │   ├── Models/
│   │   ├── Interfaces/
│   │   └── Services/
│   ├── RpgMaster.Data/
│   │   ├── InMemory/
│   │   └── SqlServer/ (Phase 3)
│   └── RpgMaster.Console/
│       ├── Program.cs
│       └── Configuration/
├── tests/
│   ├── RpgMaster.Core.Tests/
│   └── RpgMaster.Integration.Tests/
└── docs/
    ├── README.md
    └── ARCHITECTURE.md
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