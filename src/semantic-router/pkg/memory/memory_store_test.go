package memory

import (
	"context"
	"testing"
)

func TestInMemoryStore_StoreAndRetrieve(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	// Store some memories
	memories := []*Memory{
		{
			Type:    MemoryTypeSemantic,
			Content: "User's budget is $50,000 for the project",
			UserID:  "user_123",
		},
		{
			Type:    MemoryTypeSemantic,
			Content: "User prefers vegetarian food",
			UserID:  "user_123",
		},
		{
			Type:    MemoryTypeEpisodic,
			Content: "On March 15, we discussed the vacation plans to Hawaii",
			UserID:  "user_123",
		},
		{
			Type:    MemoryTypeSemantic,
			Content: "Different user's data",
			UserID:  "user_456", // Different user
		},
	}

	for _, mem := range memories {
		err := store.Store(ctx, mem)
		if err != nil {
			t.Fatalf("Store failed: %v", err)
		}
	}

	// Verify count
	if store.Count() != 4 {
		t.Errorf("Expected 4 memories, got %d", store.Count())
	}

	// Test retrieval - query about budget
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:  "What is my budget?",
		UserID: "user_123",
		Limit:  5,
	})
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected at least one result for 'budget' query")
	}

	// The budget memory should be found
	found := false
	for _, r := range results {
		if r.Memory.Content == "User's budget is $50,000 for the project" {
			found = true
			t.Logf("Found budget memory with relevance: %.2f", r.Relevance)
		}
	}
	if !found {
		t.Error("Budget memory not found in results")
	}

	// Test user isolation - user_456 should not see user_123's memories
	results, err = store.Retrieve(ctx, RetrieveOptions{
		Query:  "budget",
		UserID: "user_456",
		Limit:  5,
	})
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	for _, r := range results {
		if r.Memory.UserID != "user_456" {
			t.Errorf("User isolation failed: got memory from user %s", r.Memory.UserID)
		}
	}
}

func TestInMemoryStore_TypeFiltering(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	// Store different types
	store.Store(ctx, &Memory{
		Type:    MemoryTypeSemantic,
		Content: "User prefers blue color",
		UserID:  "user_123",
	})
	store.Store(ctx, &Memory{
		Type:    MemoryTypeEpisodic,
		Content: "User mentioned blue sky yesterday",
		UserID:  "user_123",
	})

	// Query only semantic memories
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:  "blue",
		UserID: "user_123",
		Types:  []MemoryType{MemoryTypeSemantic},
		Limit:  5,
	})
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 semantic memory, got %d", len(results))
	}

	if results[0].Memory.Type != MemoryTypeSemantic {
		t.Errorf("Expected semantic type, got %s", results[0].Memory.Type)
	}
}

func TestInMemoryStore_Forget(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	mem := &Memory{
		Type:    MemoryTypeSemantic,
		Content: "Test memory",
		UserID:  "user_123",
	}
	store.Store(ctx, mem)

	// Verify it exists
	_, err := store.Get(ctx, mem.ID)
	if err != nil {
		t.Fatalf("Memory should exist: %v", err)
	}

	// Forget it
	err = store.Forget(ctx, mem.ID)
	if err != nil {
		t.Fatalf("Forget failed: %v", err)
	}

	// Verify it's gone
	_, err = store.Get(ctx, mem.ID)
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestInMemoryStore_ForgetByScope(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	// Store memories for different users and projects
	store.Store(ctx, &Memory{Type: MemoryTypeSemantic, Content: "User1 Project1", UserID: "user_1", ProjectID: "proj_1"})
	store.Store(ctx, &Memory{Type: MemoryTypeSemantic, Content: "User1 Project2", UserID: "user_1", ProjectID: "proj_2"})
	store.Store(ctx, &Memory{Type: MemoryTypeEpisodic, Content: "User1 Episodic", UserID: "user_1", ProjectID: "proj_1"})
	store.Store(ctx, &Memory{Type: MemoryTypeSemantic, Content: "User2 Project1", UserID: "user_2", ProjectID: "proj_1"})

	if store.Count() != 4 {
		t.Fatalf("Expected 4 memories, got %d", store.Count())
	}

	// Forget all memories for user_1, proj_1, semantic type
	err := store.ForgetByScope(ctx, MemoryScope{
		UserID:    "user_1",
		ProjectID: "proj_1",
		Type:      MemoryTypeSemantic,
	})
	if err != nil {
		t.Fatalf("ForgetByScope failed: %v", err)
	}

	// Should have removed only "User1 Project1" (semantic)
	if store.Count() != 3 {
		t.Errorf("Expected 3 memories after forget, got %d", store.Count())
	}

	// Forget all for user_1
	err = store.ForgetByScope(ctx, MemoryScope{UserID: "user_1"})
	if err != nil {
		t.Fatalf("ForgetByScope failed: %v", err)
	}

	// Should have only user_2's memory left
	if store.Count() != 1 {
		t.Errorf("Expected 1 memory after forget, got %d", store.Count())
	}
}

func TestInMemoryStore_RequiresUserID(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	// Store without user_id should fail
	err := store.Store(ctx, &Memory{
		Content: "No user ID",
	})
	if err != ErrInvalidUserID {
		t.Errorf("Expected ErrInvalidUserID, got %v", err)
	}

	// Retrieve without user_id should fail
	_, err = store.Retrieve(ctx, RetrieveOptions{
		Query: "test",
	})
	if err != ErrInvalidUserID {
		t.Errorf("Expected ErrInvalidUserID, got %v", err)
	}
}
