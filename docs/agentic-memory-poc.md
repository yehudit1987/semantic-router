# Agentic Memory POC: Complete Design Document

## Executive Summary

This document describes a complete **Agentic Memory** system for the Semantic Router. Agentic Memory enables AI agents to **remember information across sessions**, providing continuity and personalization.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Memory Retrieval** | Context-aware search with pre-filtering |
| **Memory Saving** | LLM-based extraction of facts and procedures |
| **Cross-Session Persistence** | Memories survive restarts via Milvus |
| **User Isolation** | Memories scoped per user |

### Key Design Principles

1. **Simple pre-filter** decides if query should search memory
2. **Context window** from history for query disambiguation
3. **LLM extracts facts** and classifies type when saving
4. **Threshold-based filtering** on search results

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Memory Types](#3-memory-types)
4. [Pipeline Integration](#4-pipeline-integration)
5. [Memory Retrieval](#5-memory-retrieval)
6. [Memory Saving](#6-memory-saving)
7. [Memory Operations](#7-memory-operations)
8. [Data Structures](#8-data-structures)
9. [API Extension](#9-api-extension)
10. [Configuration](#10-configuration)
11. [Implementation Plan](#11-implementation-plan)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. Problem Statement

### Current State

The Response API provides conversation chaining via `previous_response_id`, but knowledge is lost across sessions:

```
Session A (March 15):
  User: "My budget for the Hawaii trip is $10,000"
  → Saved in session chain

Session B (March 20) - NEW SESSION:
  User: "What's my budget for the trip?"
  → No previous_response_id → Knowledge LOST ❌
```

### Desired State

With Agentic Memory:

```
Session A (March 15):
  User: "My budget for the Hawaii trip is $10,000"
  → Extracted and saved to Milvus

Session B (March 20) - NEW SESSION:
  User: "What's my budget for the trip?"
  → Pre-filter: memory-relevant ✓
  → Search Milvus → Found: "budget for Hawaii is $10K"
  → Inject into LLM context
  → Assistant: "Your budget for the Hawaii trip is $10,000!" ✅
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AGENTIC MEMORY ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ExtProc Pipeline                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  Request → Fact? → Tool? → Security → Cache → MEMORY → LLM      │  │
│  │              │       │                          ↑↓               │  │
│  │              └───────┴──── signals used ────────┘                │  │
│  │                                                                   │  │
│  │  Response ← [extract & store] ←─────────────────┘                │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                          │                              │
│                    ┌─────────────────────┴─────────────────────┐       │
│                    │                                           │       │
│          ┌─────────▼─────────┐                    ┌────────────▼───┐   │
│          │ Memory Retrieval  │                    │ Memory Saving  │   │
│          │  (request phase)  │                    │ (response phase)│   │
│          ├───────────────────┤                    ├────────────────┤   │
│          │ 1. Check signals  │                    │ 1. LLM extract │   │
│          │    (Fact? Tool?)  │                    │ 2. Classify    │   │
│          │ 2. Build context  │                    │ 3. Deduplicate │   │
│          │ 3. Milvus search  │                    │ 4. Store       │   │
│          │ 4. Inject to LLM  │                    │                │   │
│          └─────────┬─────────┘                    └────────┬───────┘   │
│                    │                                       │           │
│                    │         ┌──────────────┐              │           │
│                    └────────►│    Milvus    │◄─────────────┘           │
│                              └──────────────┘                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|---------------|----------|
| **Memory Filter** | Decision + search + inject | `pkg/extproc/req_filter_memory.go` |
| **Memory Extractor** | LLM-based fact extraction | `pkg/memory/extractor.go` (new) |
| **Memory Store** | Storage interface | `pkg/memory/store.go` |
| **Milvus Store** | Vector database backend | `pkg/memory/milvus_store.go` |
| **Existing Classifiers** | Fact/Tool signals (reused) | `pkg/extproc/processor_req_body.go` |

### Storage Architecture

[Issue #808](https://github.com/vllm-project/semantic-router/issues/808) suggests a multi-layer storage architecture. We implement this incrementally:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STORAGE ARCHITECTURE (Phased)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1 (MVP)                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Milvus (Vector Index)                                   │    │   │
│  │  │  • Semantic search over memories                         │    │   │
│  │  │  • Embedding storage                                     │    │   │
│  │  │  • Content + metadata                                    │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2 (Performance)                                          │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │  Redis (Hot Cache)                                       │    │   │
│  │  │  • Fast metadata lookup                                  │    │   │
│  │  │  • Recently accessed memories                            │    │   │
│  │  │  • TTL/expiration support                                │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3+ (If Needed)                                           │   │
│  │  ┌───────────────────────┐  ┌───────────────────────┐          │   │
│  │  │  Graph Store (Neo4j)  │  │  Time-Series Index    │          │   │
│  │  │  • Memory links       │  │  • Temporal queries   │          │   │
│  │  │  • Relationships      │  │  • Decay scoring      │          │   │
│  │  └───────────────────────┘  └───────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Layer | Purpose | When Needed | Status |
|-------|---------|-------------|--------|
| **Milvus** | Semantic vector search | Core functionality | ✅ MVP |
| **Redis** | Hot cache, fast access, TTL | Performance optimization | 🔶 Phase 2 |
| **Graph (Neo4j)** | Memory relationships | Multi-hop reasoning queries | ⚪ If needed |
| **Time-Series** | Temporal queries, decay | Importance scoring by time | ⚪ If needed |

> **Design Decision:** We start with Milvus only. Additional layers are added based on demonstrated need, not speculation. The `Store` interface abstracts storage, allowing backends to be added without changing retrieval/saving logic.

---

## 3. Memory Types

| Type | Purpose | Example | Status |
|------|---------|---------|--------|
| **Semantic** | Facts, preferences, knowledge | "User's budget for Hawaii is $10,000" | ✅ MVP |
| **Procedural** | How-to, steps, processes | "To deploy payment-service: run npm build, then docker push" | ✅ MVP |
| **Episodic** | Session summaries, past events | "On Dec 29 2024, user planned Hawaii vacation with $10K budget" | ✅ MVP |
| **Reflective** | Self-analysis, lessons learned | "Previous budget response was incomplete - user prefers detailed breakdowns" | 🔮 Future |

> **🔮 Reflective Memory:** Self-analysis and lessons learned. Not in scope for this POC. See [Appendix A](#appendix-a-reflective-memory).

### Memory Vector Space

Memories cluster by **content/topic**, not by type. Type is metadata:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      MEMORY VECTOR SPACE                                │
│                                                                         │
│     ┌─────────────────┐                    ┌─────────────────┐         │
│     │  BUDGET/MONEY   │                    │   DEPLOYMENT    │         │
│     │    CLUSTER      │                    │    CLUSTER      │         │
│     │                 │                    │                 │         │
│     │ ● budget=$10K   │                    │ ● npm build     │         │
│     │   (semantic)    │                    │   (procedural)  │         │
│     │ ● cost=$5K      │                    │ ● docker push   │         │
│     │   (semantic)    │                    │   (procedural)  │         │
│     └─────────────────┘                    └─────────────────┘         │
│                                                                         │
│  ● = memory with type as metadata                                      │
│  Query matches content → type comes from matched memory                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Integration

### Current Pipeline (main branch)

```
1. Response API Translation
2. Parse Request
3. Fact-Check Classification
4. Tool Detection
5. Decision & Model Selection
6. Security Checks
7. PII Detection
8. Semantic Cache Check
9. Model Routing → LLM
```

### Enhanced Pipeline with Agentic Memory

```
REQUEST PHASE:
─────────────
1.  Response API Translation
2.  Parse Request
3.  Fact-Check Classification        ──┐
4.  Tool Detection                     ├── Existing signals
5.  Decision & Model Selection       ──┘
6.  Security Checks
7.  PII Detection
8.  Semantic Cache Check ───► if HIT → return cached
9.  🆕 Memory Decision: 
    └── if (NOT Fact) AND (NOT Tool) AND (NOT Greeting) → continue
    └── else → skip to step 12
10. 🆕 Build context + rewrite query
11. 🆕 Search Milvus, inject memories
12. Model Routing → LLM

RESPONSE PHASE:
──────────────
13. Parse LLM Response
14. Cache Update
15. 🆕 Memory Extraction (async, if auto_store enabled)
16. Response API Translation
17. Return to Client
```

> **Step 10 details:** Query rewriting strategies (context prepend, LLM rewrite, HyDE) are explained in [Appendix C](#appendix-c-query-rewriting-for-memory-search).

---

## 5. Memory Retrieval

### Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEMORY RETRIEVAL FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. MEMORY DECISION (reuse existing pipeline signals)                  │
│     ──────────────────────────────────────────────────                  │
│                                                                         │
│     Pipeline already classified:                                        │
│     ├── ctx.IsFact       (Fact-Check classifier)                       │
│     ├── ctx.RequiresTool (Tool Detection)                              │
│     └── isGreeting(query) (simple pattern)                             │
│                                                                         │
│     Decision:                                                           │
│     ├── Fact query?     → SKIP (general knowledge)                     │
│     ├── Tool query?     → SKIP (tool provides answer)                  │
│     ├── Greeting?       → SKIP (no context needed)                     │
│     └── Otherwise       → SEARCH MEMORY                                │
│                                                                         │
│  2. BUILD CONTEXT + REWRITE QUERY                                       │
│     ─────────────────────────────                                       │
│     History: ["Planning vacation", "Hawaii sounds nice"]               │
│     Query: "How much?"                                                  │
│                                                                         │
│     Option A (MVP): Context prepend                                    │
│     → "How much? Hawaii vacation planning"                             │
│                                                                         │
│     Option B (v1): LLM rewrite                                         │
│     → "What is the budget for the Hawaii vacation?"                    │
│                                                                         │
│  3. MILVUS SEARCH                                                       │
│     ─────────────                                                       │
│     Embed context → Search with user_id filter → Top-k results         │
│                                                                         │
│  4. THRESHOLD FILTER                                                    │
│     ────────────────                                                    │
│     Keep only results with similarity > 0.6                            │
│                                                                         │
│  5. INJECT INTO LLM CONTEXT                                             │
│     ────────────────────────                                            │
│     Add as system message: "User's relevant context: ..."              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

#### MemoryFilter Struct

```go
// pkg/extproc/req_filter_memory.go

type MemoryFilter struct {
    store memory.Store  // Interface - can be MilvusStore or InMemoryStore
}

func NewMemoryFilter(store memory.Store) *MemoryFilter {
    return &MemoryFilter{store: store}
}
```

> **Note:** `store` is the `Store` interface (Section 8), not a specific implementation.
> At runtime, this is typically `MilvusStore` for production or `InMemoryStore` for testing.

#### Memory Decision (Reuses Existing Pipeline)

```go
// pkg/extproc/req_filter_memory.go

// shouldSearchMemory decides if query should trigger memory search
// Reuses existing pipeline classification signals
func shouldSearchMemory(ctx *RequestContext, query string) bool {
    // 1. Fact query → skip (general knowledge, not personal)
    if ctx.IsFact {
        logging.Debug("Memory: Skipping - fact query")
        return false
    }
    
    // 2. Tool required → skip (tool provides answer)
    if ctx.RequiresTool {
        logging.Debug("Memory: Skipping - tool query")
        return false
    }
    
    // 3. Greeting/social → skip (no context needed)
    if isGreeting(query) {
        logging.Debug("Memory: Skipping - greeting")
        return false
    }
    
    // 4. Default: search memory (conservative - don't miss context)
    return true
}

func isGreeting(query string) bool {
    greetings := []string{
        `^(hi|hello|hey|howdy)[\s\!\.\?]*$`,
        `^(thanks|thank you|thx)[\s\!\.\?]*$`,
        `^(bye|goodbye|see you)[\s\!\.\?]*$`,
        `^(ok|okay|sure|yes|no)[\s\!\.\?]*$`,
    }
    lower := strings.ToLower(strings.TrimSpace(query))
    for _, p := range greetings {
        if regexp.MustCompile(p).MatchString(lower) {
            return true
        }
    }
    return false
}
```

#### Context Building

```go
// buildSearchQuery builds an effective search query from history + current query
// MVP: context prepend, v1: LLM rewrite for vague queries
func buildSearchQuery(history []Message, query string) string {
    // If query is self-contained, use as-is
    if isSelfContained(query) {
        return query
    }
    
    // MVP: Simple context prepend
    context := summarizeHistory(history)
    return query + " " + context
    
    // v1 (future): LLM rewrite for vague queries
    // if isVague(query) {
    //     return rewriteWithLLM(history, query)
    // }
}

func isSelfContained(query string) bool {
    // Self-contained: "What's my budget for the Hawaii trip?"
    // NOT self-contained: "How much?", "And that one?", "What about it?"
    
    vaguePatterns := []string{`^how much\??$`, `^what about`, `^and that`, `^this one`}
    for _, p := range vaguePatterns {
        if regexp.MustCompile(`(?i)`+p).MatchString(query) {
            return false
        }
    }
    return len(query) > 20 // Short queries are often vague
}

func summarizeHistory(history []Message) string {
    // Extract key terms from last 3 user messages
    var terms []string
    count := 0
    for i := len(history) - 1; i >= 0 && count < 3; i-- {
        if history[i].Role == "user" {
            terms = append(terms, extractKeyTerms(history[i].Content))
            count++
        }
    }
    return strings.Join(terms, " ")
}

// v1: LLM-based query rewriting (future enhancement)
func rewriteWithLLM(history []Message, query string) string {
    prompt := fmt.Sprintf(`Conversation context: %s
    
Rewrite this vague query to be self-contained: "%s"
Return ONLY the rewritten query.`, summarizeHistory(history), query)
    
    // Call LLM endpoint
    resp, _ := http.Post(llmEndpoint+"/v1/chat/completions", ...)
    return parseResponse(resp)
    // "how much?" → "What is the budget for the Hawaii vacation?"
}
```

#### Full Retrieval

```go
// pkg/extproc/req_filter_memory.go

func (f *MemoryFilter) RetrieveMemories(
    ctx context.Context,
    query string,
    userID string,
    history []Message,
) ([]*memory.RetrieveResult, error) {
    
    // 1. Memory decision (skip if fact/tool/greeting)
    if !shouldSearchMemory(ctx, query) {
        logging.Debug("Memory: Skipping - not memory-relevant")
        return nil, nil
    }
    
    // 2. Build search query (context prepend or LLM rewrite)
    searchQuery := buildSearchQuery(history, query)
    
    // 3. Search Milvus
    results, err := f.store.Retrieve(ctx, memory.RetrieveOptions{
        Query:     searchQuery,
        UserID:    userID,
        Limit:     5,
        Threshold: 0.6,
    })
    if err != nil {
        return nil, err
    }
    
    logging.Infof("Memory: Retrieved %d memories", len(results))
    return results, nil
}

// InjectMemories adds memories to the LLM request
func (f *MemoryFilter) InjectMemories(
    requestBody []byte,
    memories []*memory.RetrieveResult,
) ([]byte, error) {
    if len(memories) == 0 {
        return requestBody, nil
    }
    
    // Format memories as context
    var sb strings.Builder
    sb.WriteString("## User's Relevant Context\n\n")
    for _, mem := range memories {
        sb.WriteString(fmt.Sprintf("- %s\n", mem.Memory.Content))
    }
    
    // Add as system message
    return injectSystemMessage(requestBody, sb.String())
}
```

---

## 6. Memory Saving

### Triggers

Memory extraction is triggered by three events:

| Trigger | Description | Status |
|---------|-------------|--------|
| **Every N turns** | Extract after every 10 turns | ✅ MVP |
| **End of session** | Create episodic summary when session ends | 🔮 Future |
| **Context drift** | Extract when topic changes significantly | 🔮 Future |

> **Note:** Session end detection and context drift detection require additional implementation.
> For MVP, we rely on the "every N turns" trigger only.

### Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MEMORY SAVING FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRIGGERS:                                                              │
│  ─────────                                                              │
│  ├── Every N turns (e.g., 10)      ← MVP                               │
│  ├── End of session                ← Future (needs detection)          │
│  └── Context drift detected        ← Future (needs detection)          │
│                                                                         │
│  Runs: Async (background) - no user latency                            │
│                                                                         │
│  1. GET BATCH                                                           │
│     ─────────                                                           │
│     Get last 10-15 turns from session                                  │
│                                                                         │
│  2. LLM EXTRACTION                                                      │
│     ──────────────                                                      │
│     Prompt: "Extract important facts. Include context.                 │
│              Return JSON: [{type, content}, ...]"                      │
│                                                                         │
│     LLM returns:                                                        │
│       [{"type": "semantic", "content": "budget for Hawaii is $10K"}]  │
│                                                                         │
│  3. DEDUPLICATION                                                       │
│     ─────────────                                                       │
│     For each extracted fact:                                            │
│     - Embed content                                                     │
│     - Search existing memories                                          │
│     - If similarity > 0.9: update existing                             │
│     - If similarity < 0.9: create new                                  │
│                                                                         │
│  4. STORE IN MILVUS                                                     │
│     ───────────────                                                     │
│     Memory { id, type, content, embedding, user_id, created_at }       │
│                                                                         │
│  5. SESSION END (future): Create episodic summary                      │
│     ─────────────────────────────────────────────                       │
│     "On Dec 29, user planned Hawaii vacation with $10K budget"         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```go
// pkg/memory/extractor.go

type MemoryExtractor struct {
    store       memory.Store  // Interface - can be MilvusStore or InMemoryStore
    llmEndpoint string        // LLM endpoint for fact extraction
    batchSize   int           // Extract every N turns (default: 10)
    turnCounts  map[string]int
    mu          sync.Mutex
}

// ProcessResponse extracts and stores memories (runs async)
// 
// Triggers (MVP: only first one implemented):
//   - Every N turns (e.g., 10)       ← MVP
//   - End of session                 ← Future: needs session end detection
//   - Context drift detected         ← Future: needs drift detection
//
func (e *MemoryExtractor) ProcessResponse(
    ctx context.Context,
    sessionID string,
    userID string,
    history []Message,
) error {
    e.mu.Lock()
    e.turnCounts[sessionID]++
    turnCount := e.turnCounts[sessionID]
    e.mu.Unlock()
    
    // MVP: Only extract every N turns
    // Future: Also trigger on session end or context drift
    if turnCount % e.batchSize != 0 {
        return nil
    }
    
    // Get recent batch
    batchStart := max(0, len(history) - e.batchSize - 5)
    batch := history[batchStart:]
    
    // LLM extraction
    extracted, err := e.extractWithLLM(batch)
    if err != nil {
        return err
    }
    
    // Store with deduplication
    for _, fact := range extracted {
        if e.isDuplicate(ctx, userID, fact.Content) {
            continue
        }
        
        mem := &Memory{
            ID:        generateID("mem"),
            Type:      fact.Type,
            Content:   fact.Content,
            UserID:    userID,
            Source:    "conversation",
            CreatedAt: time.Now(),
        }
        
        if err := e.store.Store(ctx, mem); err != nil {
            logging.Warnf("Failed to store memory: %v", err)
        }
    }
    
    return nil
}

// extractWithLLM uses LLM to extract facts
func (e *MemoryExtractor) extractWithLLM(messages []Message) ([]ExtractedFact, error) {
    prompt := `Extract important information from these messages.

IMPORTANT: Include CONTEXT for each fact.

For each piece of information:
- Type: "semantic" (facts, preferences) or "procedural" (instructions, how-to)
- Content: The fact WITH its context

BAD:  {"type": "semantic", "content": "budget is $10,000"}
GOOD: {"type": "semantic", "content": "budget for Hawaii vacation is $10,000"}

Messages:
` + formatMessages(messages) + `

Return JSON array (empty if nothing to remember):
[{"type": "semantic|procedural", "content": "fact with context"}]`

    // Call LLM
    reqBody := map[string]interface{}{
        "model": "qwen3",
        "messages": []map[string]string{
            {"role": "user", "content": prompt},
        },
    }
    jsonBody, _ := json.Marshal(reqBody)
    
    resp, err := http.Post(
        e.llmEndpoint+"/v1/chat/completions",
        "application/json",
        bytes.NewReader(jsonBody),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    return parseExtractedFacts(resp.Body)
}
```

---

## 7. Memory Operations

All operations that can be performed on memories. Implemented in the `Store` interface (see [Section 8](#8-data-structures)).

| Operation | Description | Trigger | Interface Method | Status |
|-----------|-------------|---------|------------------|--------|
| **Store** | Save new memory to Milvus | Auto (LLM extraction) or explicit API | `Store()` | ✅ MVP |
| **Retrieve** | Semantic search for relevant memories | Auto (on query) | `Retrieve()` | ✅ MVP |
| **Update** | Modify existing memory content | Deduplication or explicit API | `Update()` | ✅ MVP |
| **Forget** | Delete specific memory by ID | Explicit API call | `Forget()` | ✅ MVP |
| **ForgetByScope** | Delete all memories for user/project | Explicit API call | `ForgetByScope()` | ✅ MVP |
| **Consolidate** | Merge related memories into summary | Scheduled / on threshold | `Consolidate()` | 🔮 Future |
| **Reflect** | Generate insights from memory patterns | Agent-initiated | `Reflect()` | 🔮 Future |

### Forget Operations

```go
// Forget single memory
DELETE /v1/memory/{memory_id}

// Forget all memories for a user
DELETE /v1/memory?user_id=user_123

// Forget all memories for a project
DELETE /v1/memory?user_id=user_123&project_id=project_abc
```

**Use Cases:**
- User requests "forget what I told you about X"
- GDPR/privacy compliance (right to be forgotten)
- Clearing outdated information

### Future: Consolidate

Merge multiple related memories into a single summary:

```
Before:
  - "Budget for Hawaii is $10,000"
  - "Added $2,000 to Hawaii budget"
  - "Final Hawaii budget is $12,000"

After consolidation:
  - "Hawaii trip budget: $12,000 (updated from initial $10,000)"
```

**Trigger options:**
- When memory count exceeds threshold
- Scheduled background job
- On session end

### Future: Reflect

Generate insights by analyzing memory patterns:

```
Input: All memories for user_123 about "deployment"

Output (Insight):
  - "User frequently deploys payment-service (12 times)"
  - "Common issue: port conflicts"
  - "Preferred approach: docker-compose"
```

**Use case:** Agent can proactively offer help based on patterns.

---

## 8. Data Structures

### Memory

```go
// pkg/memory/types.go

type MemoryType string

const (
    MemoryTypeEpisodic   MemoryType = "episodic"
    MemoryTypeSemantic   MemoryType = "semantic"
    MemoryTypeProcedural MemoryType = "procedural"
)

type Memory struct {
    ID          string         `json:"id"`
    Type        MemoryType     `json:"type"`
    Content     string         `json:"content"`
    Embedding   []float32      `json:"-"`
    UserID      string         `json:"user_id"`
    ProjectID   string         `json:"project_id,omitempty"`
    Source      string         `json:"source,omitempty"`
    CreatedAt   time.Time      `json:"created_at"`
    AccessCount int            `json:"access_count"`
    Importance  float32        `json:"importance"`
}
```

### Store Interface

```go
// pkg/memory/store.go

type Store interface {
    // MVP Operations
    Store(ctx context.Context, memory *Memory) error                         // Save new memory
    Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) // Semantic search
    Get(ctx context.Context, id string) (*Memory, error)                     // Get by ID
    Update(ctx context.Context, id string, memory *Memory) error             // Modify existing
    Forget(ctx context.Context, id string) error                             // Delete by ID
    ForgetByScope(ctx context.Context, scope MemoryScope) error              // Delete by scope
    
    // Utility
    IsEnabled() bool
    Close() error
    
    // Future Operations (not yet implemented)
    // Consolidate(ctx context.Context, memoryIDs []string) (*Memory, error)  // Merge memories
    // Reflect(ctx context.Context, scope MemoryScope) ([]*Insight, error)    // Generate insights
}
```

---

## 9. API Extension

### Request (existing)

```go
// pkg/responseapi/types.go

type ResponseAPIRequest struct {
    // ... existing fields ...
    MemoryConfig  *MemoryConfig  `json:"memory_config,omitempty"`
    MemoryContext *MemoryContext `json:"memory_context,omitempty"`
}

type MemoryConfig struct {
    Enabled             bool     `json:"enabled"`
    MemoryTypes         []string `json:"memory_types,omitempty"`
    RetrievalLimit      int      `json:"retrieval_limit,omitempty"`
    SimilarityThreshold float32  `json:"similarity_threshold,omitempty"`
    AutoStore           bool     `json:"auto_store,omitempty"`
}

type MemoryContext struct {
    UserID    string `json:"user_id"`
    ProjectID string `json:"project_id,omitempty"`
}
```

### Example Request

```json
{
    "model": "qwen3",
    "input": "What's my budget for the trip?",
    "previous_response_id": "resp_abc123",
    "memory_config": {
        "enabled": true,
        "auto_store": true
    },
    "memory_context": {
        "user_id": "user_456"
    }
}
```

---

## 10. Configuration

```yaml
# config.yaml
memory:
  enabled: true
  store_backend: "milvus"  # or "memory" for development
  
  milvus:
    address: "milvus:19530"
    collection: "agentic_memory"
    ef_construction: 256
    m: 16
    ef: 64
  
  # Retrieval settings
  default_retrieval_limit: 5
  default_similarity_threshold: 0.6
  
  # Extraction settings (for saving)
  extraction:
    enabled: true
    batch_size: 10
    llm_endpoint: "http://qwen:8000"
```

---

## 11. Implementation Plan

### Phase 1: Retrieval

| Task | Files |
|------|-------|
| Memory decision (use existing Fact/Tool signals) | `pkg/extproc/req_filter_memory.go` |
| Context building from history | `pkg/extproc/req_filter_memory.go` |
| Milvus search + threshold filter | `pkg/memory/milvus_store.go` |
| Memory injection into request | `pkg/extproc/req_filter_memory.go` |
| Integrate in request phase | `pkg/extproc/processor_req_body.go` |

### Phase 2: Saving

| Task | Files |
|------|-------|
| Create MemoryExtractor | `pkg/memory/extractor.go` |
| LLM-based fact extraction | `pkg/memory/extractor.go` |
| Deduplication logic | `pkg/memory/extractor.go` |
| Integrate in response phase (async) | `pkg/extproc/processor_res_body.go` |

### Phase 3: Testing & Tuning

| Task | Description |
|------|-------------|
| Unit tests | Memory decision, extraction, retrieval |
| Integration tests | End-to-end flow |
| Threshold tuning | Adjust similarity threshold based on results |

---

## 12. Future Enhancements

### Phase 2: Saving Triggers

| Feature | Description | Approach |
|---------|-------------|----------|
| **Session end detection** | Trigger extraction when session ends | Timeout / explicit signal / API call |
| **Context drift detection** | Trigger when topic changes significantly | Embedding similarity between turns |

### Phase 2: Storage Layer

| Feature | Description | Priority |
|---------|-------------|----------|
| **Redis hot cache** | Fast access layer before Milvus | High |
| **TTL & expiration** | Auto-delete old memories (Redis native) | High |

### Phase 3+: Advanced Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Memory type routing** | Search specific types (semantic/procedural/episodic) | Medium |
| **Per-user quotas** | Limit storage per user | Medium |
| **Importance scoring** | LLM rates memory importance when saving | Low |
| **Graph store** | Memory relationships for multi-hop queries | If needed |
| **Time-series index** | Temporal queries and decay scoring | If needed |

---

## Appendices

### Appendix A: Reflective Memory

**Status:** Future extension - not in scope for this POC.

Self-analysis and lessons learned from past interactions. Inspired by the [Reflexion paper](https://arxiv.org/abs/2303.11366).

**What it stores:**
- Insights from incorrect or suboptimal responses
- Learned preferences about response style
- Patterns that improve future interactions

**Examples:**
- "I gave incorrect deployment steps - next time verify k8s version first"
- "User prefers bullet points over paragraphs for technical content"
- "Budget questions should include breakdown, not just total"

**Why Future:** Requires the ability to evaluate response quality and generate self-reflections, which builds on top of the core memory infrastructure.

---

### Appendix B: File Tree

```
pkg/
├── extproc/
│   ├── processor_req_body.go     (EXTEND) Integrate retrieval
│   ├── processor_res_body.go     (EXTEND) Integrate extraction
│   └── req_filter_memory.go      (EXTEND) Pre-filter, retrieval, injection
│
├── memory/
│   ├── extractor.go              (NEW) LLM-based fact extraction
│   ├── store.go                  (existing) Store interface
│   ├── milvus_store.go           (existing) Milvus implementation
│   └── types.go                  (existing) Memory types
│
├── responseapi/
│   └── types.go                  (existing) MemoryConfig, MemoryContext
│
└── config/
    └── config.go                 (EXTEND) Add extraction config
```

---

### Appendix C: Query Rewriting for Memory Search

When searching memories, vague queries like "how much?" need context to be effective. This appendix covers query rewriting strategies.

#### The Problem

```
History: ["Planning Hawaii vacation", "Looking at hotels"]
Query: "How much?"
→ Direct search for "How much?" won't find "Hawaii budget is $10,000"
```

#### Option 1: Context Prepend (MVP)

Simple concatenation - no LLM call, ~0ms latency.

```go
func buildSearchQuery(history []Message, query string) string {
    context := extractKeyTerms(history)  // "Hawaii vacation planning"
    return query + " " + context         // "How much? Hawaii vacation planning"
}
```

**Pros:** Fast, simple  
**Cons:** May include irrelevant terms

#### Option 2: LLM Query Rewriting

Use LLM to rewrite query as self-contained question. ~100-200ms latency.

```go
func rewriteQuery(history []Message, query string) string {
    prompt := `Given conversation about: %s
    Rewrite this query to be self-contained: "%s"
    Return ONLY the rewritten query.`
    return llm.Complete(fmt.Sprintf(prompt, summarize(history), query))
}
// "How much?" → "What is the budget for the Hawaii vacation?"
```

**Pros:** Natural queries, better embedding match  
**Cons:** LLM latency, cost

#### Option 3: HyDE (Hypothetical Document Embeddings)

Generate hypothetical answer, embed that instead of query.

**The Problem HyDE Solves:**
```
Query: "What's the cost?"           → embeds as QUESTION style
Stored: "Budget is $10,000"         → embeds as STATEMENT style
Result: Low similarity (style mismatch)

With HyDE:
Query → LLM generates: "The cost is approximately $10,000"
This embeds as STATEMENT style → matches stored memory!
```

```go
func hydeRewrite(query string, history []Message) string {
    prompt := `Based on this conversation: %s
    Write a short factual answer to: "%s"`
    return llm.Complete(fmt.Sprintf(prompt, summarize(history), query))
}
// "How much?" → "The budget for the Hawaii trip is approximately $10,000"
```

**Pros:** Best retrieval quality (bridges question-to-document style gap)  
**Cons:** Highest latency (~200ms), LLM cost

#### Recommendation

| Phase | Approach | Use When |
|-------|----------|----------|
| **MVP** | Context prepend | All queries (default) |
| **v1** | LLM rewrite | Vague queries ("how much?", "and that?") |
| **v2** | HyDE | **After observing** low retrieval scores for question-style queries |

> **Note:** HyDE is an optimization based on observed performance, not a prediction.
> Apply it when you see relevant memories exist but aren't being retrieved.

#### References

**Query Rewriting:**
1. **HyDE** - [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (Gao et al., 2022) - Style bridging (question → document style)
2. **RRR** - [Query Rewriting for Retrieval-Augmented LLMs](https://arxiv.org/abs/2305.14283) (Ma et al., 2023) - Trainable rewriter with RL, handles conversational context

**Agentic Memory (from [Issue #808](https://github.com/vllm-project/semantic-router/issues/808)):**

5. **MemGPT** - [Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)
6. **Generative Agents** - [Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (Park et al., 2023)
7. **Reflexion** - [Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
8. **Voyager** - [An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291) (Wang et al., 2023)

---

*Document Author: [Yehudit Kerido, Marina Koushnir]*  
*Last Updated: December 2025*  
*Status: POC DESIGN - v2 (Simplified)*  
*Based on: [Issue #808 - Explore Agentic Memory in Response API](https://github.com/vllm-project/semantic-router/issues/808)*

