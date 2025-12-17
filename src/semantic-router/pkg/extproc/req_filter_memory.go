//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// MemoryFilter handles agentic memory operations for requests.
// It retrieves relevant memories and injects them into the context,
// and stores new memories from responses.
//
// This filter integrates with the Response API to provide cross-session memory.
type MemoryFilter struct {
	store   memory.Store
	enabled bool
}

// NewMemoryFilter creates a new memory filter.
func NewMemoryFilter(store memory.Store) *MemoryFilter {
	return &MemoryFilter{
		store:   store,
		enabled: store != nil,
	}
}

// IsEnabled returns whether memory operations are enabled.
func (f *MemoryFilter) IsEnabled() bool {
	return f.enabled
}

// MemoryFilterContext holds context for memory operations during request processing.
type MemoryFilterContext struct {
	// Enabled indicates memory is enabled for this request
	Enabled bool

	// Config from the Response API request
	Config *responseapi.MemoryConfig

	// Context from the Response API request
	Context *responseapi.MemoryContext

	// RetrievedMemories are the memories found for this request
	RetrievedMemories []*memory.RetrieveResult

	// InjectedContext is the formatted memory context injected into the prompt
	InjectedContext string

	// ShouldStore indicates if the response should be stored as memory
	ShouldStore bool

	// StoredMemoryID is set after storing a memory
	StoredMemoryID string
}

// ProcessResponseAPIRequest processes a Response API request for memory operations.
// It reads memory config from the request, retrieves relevant memories,
// and returns context for injection.
func (f *MemoryFilter) ProcessResponseAPIRequest(ctx context.Context, req *responseapi.ResponseAPIRequest, userQuery string) (*MemoryFilterContext, error) {
	if !f.enabled {
		return nil, nil
	}

	// Check if memory is configured for this request
	if req.MemoryConfig == nil || !req.MemoryConfig.Enabled {
		return nil, nil
	}

	// Validate required context
	if req.MemoryContext == nil || req.MemoryContext.UserID == "" {
		logging.Warnf("Memory: memory_config.enabled=true but memory_context.user_id is missing")
		return nil, nil
	}

	memCtx := &MemoryFilterContext{
		Enabled:     true,
		Config:      req.MemoryConfig,
		Context:     req.MemoryContext,
		ShouldStore: req.MemoryConfig.AutoStore,
	}

	// Build retrieve options
	opts := memory.RetrieveOptions{
		Query:     userQuery,
		UserID:    req.MemoryContext.UserID,
		ProjectID: req.MemoryContext.ProjectID,
		Limit:     req.MemoryConfig.RetrievalLimit,
		Threshold: req.MemoryConfig.SimilarityThreshold,
	}

	// Convert memory type strings to MemoryType
	if len(req.MemoryConfig.MemoryTypes) > 0 {
		for _, typeStr := range req.MemoryConfig.MemoryTypes {
			opts.Types = append(opts.Types, memory.MemoryType(typeStr))
		}
	}

	// Set defaults
	if opts.Limit <= 0 {
		opts.Limit = 5
	}
	if opts.Threshold <= 0 {
		// Use a low default threshold - cosine similarity scores can be low
		// for semantically related but not identical content
		// TODO: debug why scores are so low
		opts.Threshold = 0.15
	}

	// Retrieve relevant memories
	results, err := f.store.Retrieve(ctx, opts)
	if err != nil {
		logging.Warnf("Memory: retrieval error: %v", err)
		// Continue without memories - don't fail the request
		return memCtx, nil
	}

	memCtx.RetrievedMemories = results

	// Format memories for context injection
	if len(results) > 0 {
		memCtx.InjectedContext = f.formatMemoriesForContext(results)
		logging.Infof("Memory: Retrieved %d memories for user %s (query: %s)",
			len(results), req.MemoryContext.UserID, truncateQuery(userQuery, 50))
	}

	return memCtx, nil
}

// formatMemoriesForContext formats retrieved memories into a context string for the LLM.
func (f *MemoryFilter) formatMemoriesForContext(memories []*memory.RetrieveResult) string {
	if len(memories) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("## Relevant Context from Memory\n\n")
	sb.WriteString("The following information was retrieved from the user's memory:\n\n")

	for i, mem := range memories {
		typeLabel := string(mem.Memory.Type)
		if typeLabel == "" {
			typeLabel = "general"
		}

		sb.WriteString(fmt.Sprintf("%d. [%s] %s\n", i+1, typeLabel, mem.Memory.Content))
	}

	sb.WriteString("\nUse this context to provide a more personalized and informed response.\n")

	return sb.String()
}

// InjectMemoryContext injects memory context into a Chat Completions request.
// It adds a system message with the retrieved memories.
// This is called after the Response API translates the request to Chat Completions format.
func (f *MemoryFilter) InjectMemoryContext(body []byte, memCtx *MemoryFilterContext) ([]byte, error) {
	if memCtx == nil || memCtx.InjectedContext == "" {
		return body, nil
	}

	// Parse the request
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, err
	}

	// Get or create messages array
	messages, ok := req["messages"].([]interface{})
	if !ok {
		return body, nil
	}

	// Create memory context message
	memoryMessage := map[string]interface{}{
		"role":    "system",
		"content": memCtx.InjectedContext,
	}

	// Insert after the first system message, or at the beginning
	insertIndex := 0
	for i, msg := range messages {
		if m, ok := msg.(map[string]interface{}); ok {
			if role, ok := m["role"].(string); ok && role == "system" {
				insertIndex = i + 1
				break
			}
		}
	}

	// Insert memory context
	newMessages := make([]interface{}, 0, len(messages)+1)
	newMessages = append(newMessages, messages[:insertIndex]...)
	newMessages = append(newMessages, memoryMessage)
	newMessages = append(newMessages, messages[insertIndex:]...)
	req["messages"] = newMessages

	return json.Marshal(req)
}

// StoreFromResponse extracts and stores memories from an LLM response.
// This is called during response processing if auto_store is enabled.
func (f *MemoryFilter) StoreFromResponse(ctx context.Context, memCtx *MemoryFilterContext, responseBody []byte, userQuery string) error {
	if !f.enabled || memCtx == nil || !memCtx.ShouldStore {
		return nil
	}

	// Parse the response to get the assistant's reply
	var resp map[string]interface{}
	if err := json.Unmarshal(responseBody, &resp); err != nil {
		return nil // Skip storage on parse error
	}

	// Extract content from choices (Chat Completions format)
	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil
	}

	content, ok := message["content"].(string)
	if !ok || content == "" {
		return nil
	}

	// Create an episodic memory of this conversation turn
	mem := &memory.Memory{
		Type:      memory.MemoryTypeEpisodic,
		Content:   fmt.Sprintf("User asked: %s\n\nAssistant replied: %s", userQuery, truncateContent(content, 1000)),
		UserID:    memCtx.Context.UserID,
		ProjectID: memCtx.Context.ProjectID,
		Metadata: map[string]interface{}{
			"source": "auto_store",
		},
		Importance: 0.5, // Default importance for auto-stored memories
	}

	if err := f.store.Store(ctx, mem); err != nil {
		logging.Warnf("Memory: failed to auto-store memory: %v", err)
		return err
	}

	memCtx.StoredMemoryID = mem.ID
	logging.Infof("Memory: Auto-stored conversation as episodic memory %s for user %s", mem.ID, memCtx.Context.UserID)
	return nil
}

// BuildMemoryOperations creates the MemoryOperations response field from the filter context.
func (f *MemoryFilter) BuildMemoryOperations(memCtx *MemoryFilterContext) *responseapi.MemoryOperations {
	if memCtx == nil || !memCtx.Enabled {
		return nil
	}

	ops := &responseapi.MemoryOperations{}

	// Add retrieved memories
	for _, result := range memCtx.RetrievedMemories {
		ops.Retrieved = append(ops.Retrieved, responseapi.RetrievedMemory{
			ID:        result.Memory.ID,
			Type:      string(result.Memory.Type),
			Content:   truncateContent(result.Memory.Content, 200),
			Relevance: result.Relevance,
		})
	}

	// Add stored memory reference
	if memCtx.StoredMemoryID != "" {
		ops.Stored = append(ops.Stored, responseapi.StoredMemoryRef{
			ID:   memCtx.StoredMemoryID,
			Type: string(memory.MemoryTypeEpisodic),
		})
	}

	// Return nil if no operations occurred
	if len(ops.Retrieved) == 0 && len(ops.Stored) == 0 {
		return nil
	}

	return ops
}

// Close releases resources held by the filter.
func (f *MemoryFilter) Close() error {
	if f.store != nil {
		return f.store.Close()
	}
	return nil
}

// truncateQuery truncates a query string for logging.
func truncateQuery(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// truncateContent truncates content for storage.
func truncateContent(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
