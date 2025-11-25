#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
success_msg() {
    echo -e "${GREEN}$1${NC}"
}

error_msg() {
    echo -e "${RED}$1${NC}"
}

info_msg() {
    echo -e "${YELLOW}$1${NC}"
}

section_header() {
    echo
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# Script header
echo
section_header "🔄 Semantic Router - Rebuild & Run"
info_msg "This script rebuilds Docker images with your latest code changes"
info_msg "and restarts the semantic-router services."
echo

# Check if we're in the project root
if [ ! -f "Makefile" ] || [ ! -f "Dockerfile.extproc" ]; then
    error_msg "❌ Error: Must be run from the semantic-router project root directory"
    exit 1
fi

# Step 1: Stop existing services
section_header "Step 1: Stopping existing services"
info_msg "Stopping Docker Compose services..."
if make docker-compose-down 2>&1 | tail -5; then
    success_msg "✅ Services stopped"
else
    info_msg "⚠️  No running services found (this is okay)"
fi

# Step 2: Build semantic-router image
section_header "Step 2: Building semantic-router image"
info_msg "Building extproc Docker image with latest code..."
if make docker-build-extproc; then
    success_msg "✅ Semantic-router image built successfully"
else
    error_msg "❌ Failed to build semantic-router image"
    exit 1
fi

# Step 3: Build dashboard image
section_header "Step 3: Building dashboard image"
info_msg "Building dashboard Docker image..."
if make docker-build-dashboard; then
    success_msg "✅ Dashboard image built successfully"
else
    error_msg "❌ Failed to build dashboard image"
    exit 1
fi

# Step 4: Build LLM Katan if needed
section_header "Step 4: Building LLM Katan image (optional)"
info_msg "Building LLM Katan test server..."
if make docker-build-llm-katan; then
    success_msg "✅ LLM Katan image built successfully"
else
    info_msg "⚠️  LLM Katan build skipped or failed (not critical)"
fi

# Step 5: Start services
section_header "Step 5: Starting services"
info_msg "Starting Docker Compose services with llm-katan profile..."
if make docker-compose-up; then
    success_msg "✅ Services started successfully"
else
    error_msg "❌ Failed to start services"
    exit 1
fi

# Step 6: Wait for services to be healthy
section_header "Step 6: Waiting for services to be healthy"
info_msg "Waiting 10 seconds for services to initialize..."
sleep 10

# Step 7: Check service status
section_header "Step 7: Service Status"
docker ps | grep -E "(semantic-router|envoy|llm-katan|dashboard)" || true

# Step 8: Send a test prompt
section_header "Step 8: Testing with a sample prompt"
info_msg "Sending test prompt: 'What is 2+2?'"
echo
curl -s -X POST http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }' | python3 -m json.tool 2>/dev/null || echo "Router is warming up..."
echo

# Final summary
echo
section_header "✅ Rebuild & Restart Complete!"
echo
success_msg "🎉 All services have been rebuilt and restarted with your latest code!"
echo
section_header "🌐 Access the Dashboard"
info_msg "Open the semantic-router dashboard in your browser:"
echo -e "  ${GREEN}http://localhost:8700${NC}"
echo
info_msg "Service endpoints:"
echo -e "  ${CYAN}• Dashboard:${NC}          http://localhost:8700"
echo -e "  ${CYAN}• Envoy Proxy:${NC}        http://localhost:8801"
echo -e "  ${CYAN}• Classification API:${NC} http://localhost:8080"
echo -e "  ${CYAN}• LLM Katan:${NC}          http://localhost:8002"
echo
info_msg "Useful commands:"
echo -e "  ${CYAN}• View logs:${NC}          docker logs -f semantic-router"
echo -e "  ${CYAN}• Stop services:${NC}      make docker-compose-down"
echo
