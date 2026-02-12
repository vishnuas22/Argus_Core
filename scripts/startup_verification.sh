#!/bin/bash
#
# Argus Core - Startup Verification Script
# Ensures all services are healthy before accepting requests
#

set -e

echo "=========================================="
echo "ARGUS CORE - STARTUP VERIFICATION"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service
check_service() {
    local service_name=$1
    local check_command=$2
    
    echo -n "Checking $service_name... "
    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed${NC}"
        return 1
    fi
}

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 5

# Check all services
echo ""
echo "=== INFRASTRUCTURE SERVICES ==="
check_service "MongoDB" "curl -s http://localhost:27017 || nc -z localhost 27017"
check_service "Redis" "redis-cli ping"
check_service "MinIO" "curl -s http://localhost:9000/minio/health/live"
check_service "Celery Worker" "supervisorctl status celery_worker | grep RUNNING"

echo ""
echo "=== APPLICATION SERVICES ==="
check_service "Backend API" "curl -sf http://localhost:8001/health"
check_service "Frontend UI" "curl -sf http://localhost:3000"

echo ""
echo "=== HEALTH CHECKS ==="

# Check backend health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8001/api/v1/health)
HEALTH_STATUS=$(echo $HEALTH_RESPONSE | jq -r '.status' 2>/dev/null || echo "unknown")

if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo -e "${GREEN}✓ Backend Health: HEALTHY${NC}"
    
    # Extract component status
    DB_STATUS=$(echo $HEALTH_RESPONSE | jq -r '.components.database' 2>/dev/null)
    STORAGE_STATUS=$(echo $HEALTH_RESPONSE | jq -r '.components.storage' 2>/dev/null)
    MODELS_LOADED=$(echo $HEALTH_RESPONSE | jq -r '.components.models.loaded' 2>/dev/null)
    
    echo "  - Database: $DB_STATUS"
    echo "  - Storage: $STORAGE_STATUS"
    echo "  - Models Loaded: $MODELS_LOADED/6"
else
    echo -e "${RED}✗ Backend Health: $HEALTH_STATUS${NC}"
fi

echo ""
echo "=== SERVICE SUMMARY ==="
supervisorctl status | grep -E "(backend|frontend|mongodb|redis|minio|celery)"

echo ""
echo "=========================================="
echo "Startup verification complete!"
echo "=========================================="
echo ""
echo "Access Points:"
echo "  - API Documentation: http://localhost:8001/docs"
echo "  - API Health: http://localhost:8001/api/v1/health"
echo "  - Frontend: http://localhost:3000"
echo ""
