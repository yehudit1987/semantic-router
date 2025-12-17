// Package memory provides agentic memory capabilities for the Semantic Router.
// This enables AI agents to remember information across sessions.
package memory

import (
	"time"
)

// MemoryType represents the type of memory
type MemoryType string

const (
	// MemoryTypeEpisodic represents past events and conversations
	// Example: "On March 15, user discussed vacation to Hawaii"
	MemoryTypeEpisodic MemoryType = "episodic"

	// MemoryTypeSemantic represents facts and preferences
	// Example: "User prefers vegetarian food"
	MemoryTypeSemantic MemoryType = "semantic"

	// MemoryTypeProcedural represents "how-to" knowledge
	// Example: "To deploy: run 'npm build' then 'docker push'"
	MemoryTypeProcedural MemoryType = "procedural"

	// MemoryTypeWorking represents temporary session context
	// Only type that uses session_id, not persisted across sessions
	MemoryTypeWorking MemoryType = "working"
)

// Memory represents a single memory entry.
// This is the core data structure stored in the MemoryStore.
type Memory struct {
	// Unique identifier (format: mem_xxxx)
	ID string `json:"id"`

	// Memory classification
	Type MemoryType `json:"type"`

	// Content
	Content   string    `json:"content"` // Human-readable content
	Embedding []float32 `json:"-"`       // Vector embedding (not serialized to JSON)

	// Scoping - determines who can access this memory
	UserID    string `json:"user_id"`              // Required: memory owner
	ProjectID string `json:"project_id,omitempty"` // Optional: project scope

	// Metadata
	Metadata   map[string]any `json:"metadata,omitempty"`
	Source     string         `json:"source,omitempty"` // "user", "assistant", "auto_store"
	Confidence float64        `json:"confidence"`       // Extraction confidence (0-1)

	// Access tracking for importance scoring
	Importance  float64   `json:"importance"`   // Computed importance score
	AccessCount int       `json:"access_count"` // Times retrieved
	CreatedAt   time.Time `json:"created_at"`
	AccessedAt  time.Time `json:"accessed_at"`
	TTL         time.Time `json:"ttl,omitempty"` // Optional expiration
}

// MemoryScope defines filtering criteria for memory operations.
// Used by ForgetByScope and similar bulk operations.
type MemoryScope struct {
	UserID    string     // Required
	ProjectID string     // Optional
	Type      MemoryType // Optional: filter by type
}

// RetrieveOptions configures memory retrieval.
type RetrieveOptions struct {
	Query     string       // Text query to embed and search
	UserID    string       // Required: scope to user
	ProjectID string       // Optional: scope to project
	Types     []MemoryType // Filter by memory types (empty = all)
	Limit     int          // Max results (default: 5)
	Threshold float32      // Min similarity score (default: 0.75)
}

// RetrieveResult represents a retrieved memory with its relevance score.
type RetrieveResult struct {
	Memory    *Memory `json:"memory"`
	Relevance float64 `json:"relevance"` // Similarity score (0-1)
}
