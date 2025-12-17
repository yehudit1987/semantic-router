package memory

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// InMemoryStore is a simple in-memory implementation of the Store interface.
// This is useful for development, testing, and as a POC before integrating Milvus.
// Note: This uses simple text matching, not semantic/vector search.
type InMemoryStore struct {
	mu       sync.RWMutex
	memories map[string]*Memory // key: memory ID
}

// NewInMemoryStore creates a new in-memory store
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		memories: make(map[string]*Memory),
	}
}

// Store saves a new memory
func (s *InMemoryStore) Store(ctx context.Context, memory *Memory) error {
	if memory.UserID == "" {
		return ErrInvalidUserID
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Generate ID if not provided
	if memory.ID == "" {
		memory.ID = "mem_" + uuid.New().String()[:8]
	}

	// Set timestamps
	now := time.Now()
	if memory.CreatedAt.IsZero() {
		memory.CreatedAt = now
	}
	memory.AccessedAt = now

	// Store a copy
	stored := *memory
	s.memories[memory.ID] = &stored

	return nil
}

// Retrieve finds memories matching the query and options.
// POC: Uses simple keyword matching. Production should use vector similarity.
func (s *InMemoryStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	if opts.UserID == "" {
		return nil, ErrInvalidUserID
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var results []*RetrieveResult
	queryLower := strings.ToLower(opts.Query)
	queryWords := strings.Fields(queryLower)

	for _, mem := range s.memories {
		// Filter by user
		if mem.UserID != opts.UserID {
			continue
		}

		// Filter by project (if specified)
		if opts.ProjectID != "" && mem.ProjectID != opts.ProjectID {
			continue
		}

		// Filter by type (if specified)
		if len(opts.Types) > 0 {
			typeMatch := false
			for _, t := range opts.Types {
				if mem.Type == t {
					typeMatch = true
					break
				}
			}
			if !typeMatch {
				continue
			}
		}

		// Simple relevance scoring based on keyword matching
		// POC: Count how many query words appear in the content
		contentLower := strings.ToLower(mem.Content)
		matchCount := 0
		for _, word := range queryWords {
			if strings.Contains(contentLower, word) {
				matchCount++
			}
		}

		// Only include if there's some match
		if matchCount > 0 {
			relevance := float64(matchCount) / float64(len(queryWords))

			// Update access tracking
			mem.AccessCount++
			mem.AccessedAt = time.Now()

			results = append(results, &RetrieveResult{
				Memory:    mem,
				Relevance: relevance,
			})
		}
	}

	// Sort by relevance (simple bubble sort for POC)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Relevance > results[i].Relevance {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Apply limit
	limit := opts.Limit
	if limit <= 0 {
		limit = 5 // default
	}
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// Get retrieves a specific memory by ID
func (s *InMemoryStore) Get(ctx context.Context, id string) (*Memory, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	mem, ok := s.memories[id]
	if !ok {
		return nil, ErrNotFound
	}

	// Return a copy
	result := *mem
	return &result, nil
}

// Update modifies an existing memory
func (s *InMemoryStore) Update(ctx context.Context, id string, memory *Memory) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.memories[id]; !ok {
		return ErrNotFound
	}

	memory.ID = id
	s.memories[id] = memory
	return nil
}

// Forget removes a memory by ID
func (s *InMemoryStore) Forget(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.memories[id]; !ok {
		return ErrNotFound
	}

	delete(s.memories, id)
	return nil
}

// ForgetByScope removes all memories matching the scope
func (s *InMemoryStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id, mem := range s.memories {
		match := true

		// Check user scope
		if scope.UserID != "" && mem.UserID != scope.UserID {
			match = false
		}

		// Check project scope
		if scope.ProjectID != "" && mem.ProjectID != scope.ProjectID {
			match = false
		}

		// Check type scope
		if scope.Type != "" && mem.Type != scope.Type {
			match = false
		}

		if match {
			delete(s.memories, id)
		}
	}
	return nil
}

// IsEnabled returns true (in-memory store is always enabled)
func (s *InMemoryStore) IsEnabled() bool {
	return true
}

// Close releases any resources (no-op for in-memory)
func (s *InMemoryStore) Close() error {
	return nil
}

// Count returns the number of stored memories (for testing)
func (s *InMemoryStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.memories)
}
