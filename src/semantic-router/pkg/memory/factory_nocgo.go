//go:build windows || !cgo

package memory

import (
	"fmt"
)

// StoreType defines the backend type for memory storage
type StoreType string

const (
	StoreTypeMemory StoreType = "memory"
	StoreTypeMilvus StoreType = "milvus"
)

// MilvusStoreConfig is a stub for non-CGO builds
type MilvusStoreConfig struct {
	Address             string
	Database            string
	CollectionName      string
	SimilarityThreshold float32
	TopK                int
	EfConstruction      int
	M                   int
	Ef                  int
}

// StoreConfig configures the memory store
type StoreConfig struct {
	Type   StoreType          `yaml:"type" json:"type"`
	Milvus *MilvusStoreConfig `yaml:"milvus,omitempty" json:"milvus,omitempty"`
}

// DefaultStoreConfig returns sensible defaults for development
func DefaultStoreConfig() *StoreConfig {
	return &StoreConfig{
		Type: StoreTypeMemory,
	}
}

// NewStore creates a memory store based on configuration
// Note: Milvus is not available without CGO
func NewStore(config *StoreConfig) (Store, error) {
	if config == nil {
		config = DefaultStoreConfig()
	}

	switch config.Type {
	case StoreTypeMemory, "":
		return NewInMemoryStore(), nil

	case StoreTypeMilvus:
		return nil, fmt.Errorf("milvus store requires CGO (build with CGO_ENABLED=1)")

	default:
		return nil, fmt.Errorf("unknown store type: %s", config.Type)
	}
}

// NewMemoryStore is a convenience function for creating an in-memory store
func NewMemoryStore() Store {
	return NewInMemoryStore()
}
