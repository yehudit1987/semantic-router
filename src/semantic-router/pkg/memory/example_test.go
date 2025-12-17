package memory_test

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// Example demonstrates basic usage of the Agentic Memory system
func Example() {
	ctx := context.Background()
	store := memory.NewInMemoryStore()

	// === MONDAY SESSION ===
	// User mentions their budget
	store.Store(ctx, &memory.Memory{
		Type:       memory.MemoryTypeSemantic,
		Content:    "User's project budget is $50,000",
		UserID:     "user_123",
		ProjectID:  "project_alpha",
		Importance: 0.9,
	})

	store.Store(ctx, &memory.Memory{
		Type:       memory.MemoryTypeSemantic,
		Content:    "User prefers Toyota cars",
		UserID:     "user_123",
		Importance: 0.7,
	})

	fmt.Println("=== Monday: Stored user preferences ===")
	fmt.Printf("Total memories: %d\n", store.Count())

	// === FRIDAY SESSION ===
	// Different session, same user asks about budget
	results, _ := store.Retrieve(ctx, memory.RetrieveOptions{
		Query:  "What is my budget?",
		UserID: "user_123",
		Limit:  5,
	})

	fmt.Println("\n=== Friday: User asks 'What is my budget?' ===")
	fmt.Printf("Found %d relevant memories:\n", len(results))
	for i, r := range results {
		fmt.Printf("  %d. [%.0f%% relevant] %s\n", i+1, r.Relevance*100, r.Memory.Content)
	}

	// === CROSS-SESSION MEMORY WORKS! ===
	// The memory from Monday is found on Friday

	// Output:
	// === Monday: Stored user preferences ===
	// Total memories: 2
	//
	// === Friday: User asks 'What is my budget?' ===
	// Found 1 relevant memories:
	//   1. [25% relevant] User's project budget is $50,000
}

// ExampleTypeFiltering demonstrates filtering by memory type
func Example_typeFiltering() {
	ctx := context.Background()
	store := memory.NewInMemoryStore()

	// Store different types
	store.Store(ctx, &memory.Memory{
		Type:    memory.MemoryTypeSemantic,
		Content: "User's favorite color is blue",
		UserID:  "user_123",
	})

	store.Store(ctx, &memory.Memory{
		Type:    memory.MemoryTypeEpisodic,
		Content: "Yesterday user mentioned liking blue skies",
		UserID:  "user_123",
	})

	store.Store(ctx, &memory.Memory{
		Type:    memory.MemoryTypeProcedural,
		Content: "To deploy: run 'npm build' then 'docker push'",
		UserID:  "user_123",
	})

	// Query only semantic memories about "blue"
	results, _ := store.Retrieve(ctx, memory.RetrieveOptions{
		Query:  "blue",
		UserID: "user_123",
		Types:  []memory.MemoryType{memory.MemoryTypeSemantic},
	})

	fmt.Println("=== Semantic memories about 'blue': ===")
	for _, r := range results {
		fmt.Printf("  - %s\n", r.Memory.Content)
	}

	// Query procedural memories about "deploy"
	results, _ = store.Retrieve(ctx, memory.RetrieveOptions{
		Query:  "deploy",
		UserID: "user_123",
		Types:  []memory.MemoryType{memory.MemoryTypeProcedural},
	})

	fmt.Println("\n=== Procedural memories about 'deploy': ===")
	for _, r := range results {
		fmt.Printf("  - %s\n", r.Memory.Content)
	}

	// Output:
	// === Semantic memories about 'blue': ===
	//   - User's favorite color is blue
	//
	// === Procedural memories about 'deploy': ===
	//   - To deploy: run 'npm build' then 'docker push'
}

// ExampleUserIsolation demonstrates that users cannot see each other's memories
func Example_userIsolation() {
	ctx := context.Background()
	store := memory.NewInMemoryStore()

	// Store memory for user A
	store.Store(ctx, &memory.Memory{
		Type:    memory.MemoryTypeSemantic,
		Content: "User A's salary is $100,000",
		UserID:  "user_A",
	})

	// Store memory for user B
	store.Store(ctx, &memory.Memory{
		Type:    memory.MemoryTypeSemantic,
		Content: "User B's salary is $80,000",
		UserID:  "user_B",
	})

	// User A queries - should only see their own data
	results, _ := store.Retrieve(ctx, memory.RetrieveOptions{
		Query:  "salary",
		UserID: "user_A",
	})

	fmt.Println("=== User A queries 'salary': ===")
	for _, r := range results {
		fmt.Printf("  - [%s] %s\n", r.Memory.UserID, r.Memory.Content)
	}

	// User B queries - should only see their own data
	results, _ = store.Retrieve(ctx, memory.RetrieveOptions{
		Query:  "salary",
		UserID: "user_B",
	})

	fmt.Println("\n=== User B queries 'salary': ===")
	for _, r := range results {
		fmt.Printf("  - [%s] %s\n", r.Memory.UserID, r.Memory.Content)
	}

	// Output:
	// === User A queries 'salary': ===
	//   - [user_A] User A's salary is $100,000
	//
	// === User B queries 'salary': ===
	//   - [user_B] User B's salary is $80,000
}
