package memory

import (
	"context"
	"errors"
)

// Common errors
var (
	ErrNotFound      = errors.New("memory not found")
	ErrInvalidUserID = errors.New("user_id is required")
)

// Store defines the interface for agentic memory storage.
// This is the main interface that different backends (in-memory, Milvus, Redis) will implement.
//
// Implementations:
//   - InMemoryStore: Development/testing (memory_store.go)
//   - MilvusStore: Production with vector search (milvus_store.go)
type Store interface {
	// Store saves a new memory with its embedding
	Store(ctx context.Context, memory *Memory) error

	// Retrieve finds memories similar to the query
	Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error)

	// Get retrieves a specific memory by ID
	Get(ctx context.Context, id string) (*Memory, error)

	// Update modifies an existing memory
	Update(ctx context.Context, id string, memory *Memory) error

	// Forget removes a memory by ID
	Forget(ctx context.Context, id string) error

	// ForgetByScope removes all memories matching the scope
	ForgetByScope(ctx context.Context, scope MemoryScope) error

	// IsEnabled returns whether the store is active
	IsEnabled() bool

	// Close releases resources
	Close() error
}

// MemoryUpdate contains fields that can be updated for an existing memory.
// Nil fields are not updated.
type MemoryUpdate struct {
	Content    *string
	Metadata   map[string]any
	Importance *float64
}
