//go:build windows || !cgo

package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// MemoryFilter is a stub for non-CGO builds.
type MemoryFilter struct {
	enabled bool
}

// NewMemoryFilter creates a disabled memory filter for non-CGO builds.
func NewMemoryFilter(store interface{}) *MemoryFilter {
	return &MemoryFilter{enabled: false}
}

// IsEnabled returns false for non-CGO builds.
func (f *MemoryFilter) IsEnabled() bool {
	return false
}

// MemoryFilterContext is a stub for non-CGO builds.
type MemoryFilterContext struct {
	Enabled           bool
	ShouldStore       bool
	StoredMemoryID    string
	InjectedContext   string
	RetrievedMemories []interface{}
}

// ProcessResponseAPIRequest is a no-op for non-CGO builds.
func (f *MemoryFilter) ProcessResponseAPIRequest(ctx context.Context, req *responseapi.ResponseAPIRequest, userQuery string) (*MemoryFilterContext, error) {
	return nil, nil
}

// InjectMemoryContext is a no-op for non-CGO builds.
func (f *MemoryFilter) InjectMemoryContext(body []byte, memCtx *MemoryFilterContext) ([]byte, error) {
	return body, nil
}

// StoreFromResponse is a no-op for non-CGO builds.
func (f *MemoryFilter) StoreFromResponse(ctx context.Context, memCtx *MemoryFilterContext, responseBody []byte, userQuery string) error {
	return nil
}

// BuildMemoryOperations is a no-op for non-CGO builds.
func (f *MemoryFilter) BuildMemoryOperations(memCtx *MemoryFilterContext) *responseapi.MemoryOperations {
	return nil
}

// Close is a no-op for non-CGO builds.
func (f *MemoryFilter) Close() error {
	return nil
}
