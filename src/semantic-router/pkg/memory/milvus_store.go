//go:build !windows && cgo

package memory

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// MilvusStoreConfig configures the Milvus memory store
type MilvusStoreConfig struct {
	// Connection
	Address  string // e.g., "localhost:19530"
	Database string // optional

	// Collection
	CollectionName string // e.g., "agentic_memory"

	// Search
	SimilarityThreshold float32 // minimum similarity (0-1), e.g., 0.7
	TopK                int     // max results to return

	// Index
	EfConstruction int // HNSW build parameter (default: 256)
	M              int // HNSW connections (default: 16)
	Ef             int // HNSW search parameter (default: 64)
}

// DefaultMilvusConfig returns sensible defaults
func DefaultMilvusConfig() *MilvusStoreConfig {
	return &MilvusStoreConfig{
		Address:             "localhost:19530",
		CollectionName:      "agentic_memory",
		SimilarityThreshold: 0.7,
		TopK:                10,
		EfConstruction:      256,
		M:                   16,
		Ef:                  64,
	}
}

// MilvusStore implements the Store interface using Milvus vector database
// for semantic similarity search across memories.
type MilvusStore struct {
	client         client.Client
	config         *MilvusStoreConfig
	collectionName string
	dimension      int // embedding dimension (auto-detected)
	mu             sync.RWMutex
}

// NewMilvusStore creates a new Milvus-backed memory store
func NewMilvusStore(config *MilvusStoreConfig) (*MilvusStore, error) {
	if config == nil {
		config = DefaultMilvusConfig()
	}

	logging.Infof("MilvusMemoryStore: connecting to %s", config.Address)

	// Connect to Milvus
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	milvusClient, err := client.NewGrpcClient(ctx, config.Address)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Milvus: %w", err)
	}

	store := &MilvusStore{
		client:         milvusClient,
		config:         config,
		collectionName: config.CollectionName,
	}

	// Initialize collection
	if err := store.initializeCollection(ctx); err != nil {
		milvusClient.Close()
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}

	logging.Infof("MilvusMemoryStore: initialized successfully with collection '%s'", config.CollectionName)
	return store, nil
}

// initializeCollection creates the memory collection if it doesn't exist
func (s *MilvusStore) initializeCollection(ctx context.Context) error {
	// Check if collection exists
	exists, err := s.client.HasCollection(ctx, s.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if exists {
		logging.Infof("MilvusMemoryStore: collection '%s' already exists", s.collectionName)
		// Load collection for searching
		if err := s.client.LoadCollection(ctx, s.collectionName, false); err != nil {
			logging.Warnf("MilvusMemoryStore: failed to load collection: %v", err)
		}
		return nil
	}

	// Detect embedding dimension
	testEmbedding, err := candle_binding.GetEmbedding("test", 0)
	if err != nil {
		return fmt.Errorf("failed to get test embedding: %w", err)
	}
	s.dimension = len(testEmbedding)
	logging.Infof("MilvusMemoryStore: detected embedding dimension: %d", s.dimension)

	// Define schema
	schema := &entity.Schema{
		CollectionName: s.collectionName,
		Description:    "Agentic Memory storage for cross-session AI memory",
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:     false,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:       "user_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "128"},
			},
			{
				Name:       "project_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "128"},
			},
			{
				Name:       "memory_type",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "32"},
			},
			{
				Name:       "content",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     "importance",
				DataType: entity.FieldTypeFloat,
			},
			{
				Name:     "created_at",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", s.dimension),
				},
			},
		},
	}

	// Create collection
	if err := s.client.CreateCollection(ctx, schema, 2); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	// Create HNSW index for vector search
	idx, err := entity.NewIndexHNSW(entity.COSINE, s.config.M, s.config.EfConstruction)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	if err := s.client.CreateIndex(ctx, s.collectionName, "embedding", idx, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	// Load collection for searching
	if err := s.client.LoadCollection(ctx, s.collectionName, false); err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	logging.Infof("MilvusMemoryStore: created collection '%s' with HNSW index", s.collectionName)
	return nil
}

// Store saves a new memory with its embedding
func (s *MilvusStore) Store(ctx context.Context, memory *Memory) error {
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

	// Generate embedding
	logging.Debugf("MilvusMemoryStore: generating embedding for content length %d", len(memory.Content))
	embedding, err := candle_binding.GetEmbedding(memory.Content, 0)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}
	logging.Infof("MilvusMemoryStore: generated embedding with dimension %d", len(embedding))
	if len(embedding) == 0 {
		contentPreview := memory.Content
		if len(contentPreview) > 100 {
			contentPreview = contentPreview[:100] + "..."
		}
		return fmt.Errorf("embedding generation returned empty vector for content: %s", contentPreview)
	}
	memory.Embedding = embedding

	// Prepare data columns
	ids := []string{memory.ID}
	userIDs := []string{memory.UserID}
	projectIDs := []string{memory.ProjectID}
	memoryTypes := []string{string(memory.Type)}
	contents := []string{memory.Content}
	importances := []float32{float32(memory.Importance)}
	createdAts := []int64{memory.CreatedAt.Unix()}
	embeddings := [][]float32{embedding}

	// Insert into Milvus
	_, err = s.client.Insert(ctx, s.collectionName, "",
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnVarChar("user_id", userIDs),
		entity.NewColumnVarChar("project_id", projectIDs),
		entity.NewColumnVarChar("memory_type", memoryTypes),
		entity.NewColumnVarChar("content", contents),
		entity.NewColumnFloat("importance", importances),
		entity.NewColumnInt64("created_at", createdAts),
		entity.NewColumnFloatVector("embedding", s.dimension, embeddings),
	)
	if err != nil {
		return fmt.Errorf("failed to insert memory: %w", err)
	}

	// Flush to ensure data is persisted
	if err := s.client.Flush(ctx, s.collectionName, false); err != nil {
		logging.Warnf("MilvusMemoryStore: flush warning: %v", err)
	}

	logging.Debugf("MilvusMemoryStore: stored memory %s for user %s", memory.ID, memory.UserID)
	return nil
}

// Retrieve finds memories matching the query using semantic similarity
func (s *MilvusStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	if opts.UserID == "" {
		return nil, ErrInvalidUserID
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Generate query embedding
	queryEmbedding, err := candle_binding.GetEmbedding(opts.Query, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Build filter expression
	filter := fmt.Sprintf("user_id == \"%s\"", opts.UserID)
	if opts.ProjectID != "" {
		filter += fmt.Sprintf(" && project_id == \"%s\"", opts.ProjectID)
	}
	if len(opts.Types) > 0 {
		typeFilter := ""
		for i, t := range opts.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", t)
		}
		filter += fmt.Sprintf(" && (%s)", typeFilter)
	}

	// Set search parameters
	topK := opts.Limit
	if topK <= 0 {
		topK = s.config.TopK
	}

	searchParam, err := entity.NewIndexHNSWSearchParam(s.config.Ef)
	if err != nil {
		return nil, fmt.Errorf("failed to create search param: %w", err)
	}

	// Execute semantic search
	searchResult, err := s.client.Search(
		ctx,
		s.collectionName,
		[]string{},
		filter,
		[]string{"id", "user_id", "project_id", "memory_type", "content", "importance", "created_at"},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		"embedding",
		entity.COSINE,
		topK,
		searchParam,
	)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		logging.Debugf("MilvusMemoryStore: search returned 0 results")
		return []*RetrieveResult{}, nil
	}

	// Extract results
	var results []*RetrieveResult
	result := searchResult[0]

	logging.Infof("MilvusMemoryStore: search returned %d raw results", result.ResultCount)

	for i := 0; i < result.ResultCount; i++ {
		score := result.Scores[i]

		// Filter by threshold
		threshold := opts.Threshold
		if threshold <= 0 {
			threshold = s.config.SimilarityThreshold
		}

		logging.Infof("MilvusMemoryStore: result %d score=%.4f, threshold=%.4f", i, score, threshold)

		if score < threshold {
			logging.Infof("MilvusMemoryStore: skipping result %d (score %.4f < threshold %.4f)", i, score, threshold)
			continue
		}

		// Extract fields
		var mem Memory
		for _, field := range result.Fields {
			switch field.Name() {
			case "id":
				col := field.(*entity.ColumnVarChar)
				mem.ID, _ = col.ValueByIdx(i)
			case "user_id":
				col := field.(*entity.ColumnVarChar)
				mem.UserID, _ = col.ValueByIdx(i)
			case "project_id":
				col := field.(*entity.ColumnVarChar)
				mem.ProjectID, _ = col.ValueByIdx(i)
			case "memory_type":
				col := field.(*entity.ColumnVarChar)
				typeStr, _ := col.ValueByIdx(i)
				mem.Type = MemoryType(typeStr)
			case "content":
				col := field.(*entity.ColumnVarChar)
				mem.Content, _ = col.ValueByIdx(i)
			case "importance":
				col := field.(*entity.ColumnFloat)
				imp, _ := col.ValueByIdx(i)
				mem.Importance = float64(imp)
			case "created_at":
				col := field.(*entity.ColumnInt64)
				ts, _ := col.ValueByIdx(i)
				mem.CreatedAt = time.Unix(ts, 0)
			}
		}

		results = append(results, &RetrieveResult{
			Memory:    &mem,
			Relevance: float64(score),
		})
	}

	logging.Debugf("MilvusMemoryStore: found %d memories for query '%s'", len(results), opts.Query)
	return results, nil
}

// Get retrieves a specific memory by ID
func (s *MilvusStore) Get(ctx context.Context, id string) (*Memory, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Query by ID
	result, err := s.client.Query(
		ctx,
		s.collectionName,
		[]string{},
		fmt.Sprintf("id == \"%s\"", id),
		[]string{"id", "user_id", "project_id", "memory_type", "content", "importance", "created_at"},
	)
	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}

	if len(result) == 0 {
		return nil, ErrNotFound
	}

	// Extract memory from result
	var mem Memory
	for _, col := range result {
		switch col.Name() {
		case "id":
			c := col.(*entity.ColumnVarChar)
			mem.ID, _ = c.ValueByIdx(0)
		case "user_id":
			c := col.(*entity.ColumnVarChar)
			mem.UserID, _ = c.ValueByIdx(0)
		case "project_id":
			c := col.(*entity.ColumnVarChar)
			mem.ProjectID, _ = c.ValueByIdx(0)
		case "memory_type":
			c := col.(*entity.ColumnVarChar)
			typeStr, _ := c.ValueByIdx(0)
			mem.Type = MemoryType(typeStr)
		case "content":
			c := col.(*entity.ColumnVarChar)
			mem.Content, _ = c.ValueByIdx(0)
		case "importance":
			c := col.(*entity.ColumnFloat)
			imp, _ := c.ValueByIdx(0)
			mem.Importance = float64(imp)
		case "created_at":
			c := col.(*entity.ColumnInt64)
			ts, _ := c.ValueByIdx(0)
			mem.CreatedAt = time.Unix(ts, 0)
		}
	}

	return &mem, nil
}

// Update modifies an existing memory
func (s *MilvusStore) Update(ctx context.Context, id string, memory *Memory) error {
	// Milvus doesn't support direct updates - delete and re-insert
	if err := s.Forget(ctx, id); err != nil && err != ErrNotFound {
		return err
	}
	memory.ID = id
	return s.Store(ctx, memory)
}

// Forget removes a memory by ID
func (s *MilvusStore) Forget(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.client.Delete(ctx, s.collectionName, "", fmt.Sprintf("id == \"%s\"", id)); err != nil {
		return fmt.Errorf("forget failed: %w", err)
	}
	return nil
}

// ForgetByScope removes all memories matching the scope
func (s *MilvusStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Build filter expression
	var filters []string
	if scope.UserID != "" {
		filters = append(filters, fmt.Sprintf("user_id == \"%s\"", scope.UserID))
	}
	if scope.ProjectID != "" {
		filters = append(filters, fmt.Sprintf("project_id == \"%s\"", scope.ProjectID))
	}
	if scope.Type != "" {
		filters = append(filters, fmt.Sprintf("memory_type == \"%s\"", scope.Type))
	}

	if len(filters) == 0 {
		return nil // No scope specified, don't delete anything
	}

	filter := filters[0]
	for i := 1; i < len(filters); i++ {
		filter += " && " + filters[i]
	}

	if err := s.client.Delete(ctx, s.collectionName, "", filter); err != nil {
		return fmt.Errorf("forget by scope failed: %w", err)
	}
	return nil
}

// IsEnabled returns whether the store is active
func (s *MilvusStore) IsEnabled() bool {
	return s.client != nil
}

// Close releases resources
func (s *MilvusStore) Close() error {
	return s.client.Close()
}
