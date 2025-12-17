//go:build !windows && cgo

package memory

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// StoreType defines the backend type for memory storage
type StoreType string

const (
	StoreTypeMemory StoreType = "memory"
	StoreTypeMilvus StoreType = "milvus"
)

// StoreConfig configures the memory store
type StoreConfig struct {
	// Type is the backend type: "memory" or "milvus"
	Type StoreType `yaml:"type" json:"type"`

	// Milvus configuration (when Type is "milvus")
	Milvus *MilvusStoreConfig `yaml:"milvus,omitempty" json:"milvus,omitempty"`
}

// DefaultStoreConfig returns sensible defaults for development
func DefaultStoreConfig() *StoreConfig {
	return &StoreConfig{
		Type: StoreTypeMemory, // Default to in-memory for easy testing
	}
}

// NewStore creates a memory store based on configuration
func NewStore(config *StoreConfig) (Store, error) {
	if config == nil {
		config = DefaultStoreConfig()
	}

	switch config.Type {
	case StoreTypeMemory, "":
		logging.Infof("MemoryStore: using in-memory backend")
		return NewInMemoryStore(), nil

	case StoreTypeMilvus:
		logging.Infof("MemoryStore: using Milvus backend")
		if config.Milvus == nil {
			config.Milvus = DefaultMilvusConfig()
		}
		return NewMilvusStore(config.Milvus)

	default:
		return nil, fmt.Errorf("unknown store type: %s", config.Type)
	}
}

// NewMemoryStore is a convenience function for creating an in-memory store
func NewMemoryStore() Store {
	return NewInMemoryStore()
}
