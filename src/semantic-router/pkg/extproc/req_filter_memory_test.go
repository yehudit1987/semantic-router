package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestMemoryFilter_ProcessResponseAPIRequest_Disabled(t *testing.T) {
	// Create filter without a store (disabled)
	filter := NewMemoryFilter(nil)

	if filter.IsEnabled() {
		t.Error("Expected filter to be disabled when store is nil")
	}

	req := &responseapi.ResponseAPIRequest{}
	memCtx, err := filter.ProcessResponseAPIRequest(context.Background(), req, "test query")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if memCtx != nil {
		t.Error("Expected nil context when filter is disabled")
	}
}

func TestMemoryFilter_InjectMemoryContext_NilContext(t *testing.T) {
	filter := NewMemoryFilter(nil)

	body := []byte(`{"messages": [{"role": "user", "content": "hello"}]}`)
	result, err := filter.InjectMemoryContext(body, nil)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Should return unchanged body
	if string(result) != string(body) {
		t.Error("Expected unchanged body when context is nil")
	}
}

func TestMemoryFilter_InjectMemoryContext_EmptyContext(t *testing.T) {
	filter := NewMemoryFilter(nil)

	body := []byte(`{"messages": [{"role": "user", "content": "hello"}]}`)
	memCtx := &MemoryFilterContext{
		Enabled:         true,
		InjectedContext: "", // Empty context
	}

	result, err := filter.InjectMemoryContext(body, memCtx)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Should return unchanged body
	if string(result) != string(body) {
		t.Error("Expected unchanged body when InjectedContext is empty")
	}
}

func TestMemoryFilter_InjectMemoryContext_WithContext(t *testing.T) {
	filter := &MemoryFilter{enabled: true} // Minimal enabled filter

	body := []byte(`{"messages": [{"role": "user", "content": "hello"}]}`)
	memCtx := &MemoryFilterContext{
		Enabled:         true,
		InjectedContext: "User's budget is $50,000",
	}

	result, err := filter.InjectMemoryContext(body, memCtx)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Parse result to verify injection
	var parsed map[string]interface{}
	if err := json.Unmarshal(result, &parsed); err != nil {
		t.Fatalf("Failed to parse result: %v", err)
	}

	messages, ok := parsed["messages"].([]interface{})
	if !ok {
		t.Fatal("Messages not found in result")
	}

	// Should have 2 messages now (injected system + original user)
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}

	// First message should be the injected system message
	firstMsg, ok := messages[0].(map[string]interface{})
	if !ok {
		t.Fatal("First message is not a map")
	}

	if firstMsg["role"] != "system" {
		t.Errorf("Expected first message role to be 'system', got '%v'", firstMsg["role"])
	}

	content, ok := firstMsg["content"].(string)
	if !ok || content != "User's budget is $50,000" {
		t.Errorf("Unexpected content: %v", firstMsg["content"])
	}
}

func TestMemoryFilter_InjectMemoryContext_AfterSystemMessage(t *testing.T) {
	filter := &MemoryFilter{enabled: true}

	// Request with existing system message
	body := []byte(`{
		"messages": [
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": "hello"}
		]
	}`)
	memCtx := &MemoryFilterContext{
		Enabled:         true,
		InjectedContext: "Memory context here",
	}

	result, err := filter.InjectMemoryContext(body, memCtx)

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(result, &parsed); err != nil {
		t.Fatalf("Failed to parse result: %v", err)
	}

	messages, ok := parsed["messages"].([]interface{})
	if !ok {
		t.Fatal("Messages not found in result")
	}

	// Should have 3 messages: original system, injected memory, original user
	if len(messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(messages))
	}

	// First should be original system
	msg0, _ := messages[0].(map[string]interface{})
	if msg0["content"] != "You are a helpful assistant" {
		t.Error("First message should be original system message")
	}

	// Second should be injected memory
	msg1, _ := messages[1].(map[string]interface{})
	if msg1["role"] != "system" || msg1["content"] != "Memory context here" {
		t.Error("Second message should be injected memory context")
	}

	// Third should be user message
	msg2, _ := messages[2].(map[string]interface{})
	if msg2["role"] != "user" {
		t.Error("Third message should be user message")
	}
}

func TestMemoryFilter_BuildMemoryOperations(t *testing.T) {
	filter := &MemoryFilter{enabled: true}

	// Test nil context
	ops := filter.BuildMemoryOperations(nil)
	if ops != nil {
		t.Error("Expected nil operations for nil context")
	}

	// Test disabled context
	ops = filter.BuildMemoryOperations(&MemoryFilterContext{Enabled: false})
	if ops != nil {
		t.Error("Expected nil operations for disabled context")
	}

	// Test empty context (no operations)
	ops = filter.BuildMemoryOperations(&MemoryFilterContext{Enabled: true})
	if ops != nil {
		t.Error("Expected nil operations when no memories retrieved or stored")
	}
}
