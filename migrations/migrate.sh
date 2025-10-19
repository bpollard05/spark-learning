 =====================================================
# Migration Management Script
# =====================================================
# scripts/migrate.sh

#!/bin/bash

# Database migration management script

set -e

COMMAND=$1
MESSAGE=$2

case $COMMAND in
  "init")
    echo "Initializing Alembic..."
    alembic init migrations
    ;;
  
  "create")
    if [ -z "$MESSAGE" ]; then
      echo "Error: Migration message required"
      echo "Usage: ./migrate.sh create 'migration message'"
      exit 1
    fi
    echo "Creating new migration: $MESSAGE"
    alembic revision --autogenerate -m "$MESSAGE"
    ;;
  
  "up")
    echo "Running migrations..."
    alembic upgrade head
    ;;
  
  "down")
    echo "Rolling back last migration..."
    alembic downgrade -1
    ;;
  
  "history")
    echo "Migration history:"
    alembic history
    ;;
  
  "current")
    echo "Current migration version:"
    alembic current
    ;;
  
  "reset")
    echo "WARNING: This will drop all tables and re-run migrations"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      alembic downgrade base
      alembic upgrade head
      echo "Database reset complete"
    else
      echo "Reset cancelled"
    fi
    ;;
  
  *)
    echo "Usage: ./migrate.sh {init|create|up|down|history|current|reset}"
    echo ""
    echo "Commands:"
    echo "  init              Initialize Alembic migrations"
    echo "  create 'message'  Create a new migration"
    echo "  up                Run all pending migrations"
    echo "  down              Rollback last migration"
    echo "  history           Show migration history"
    echo "  current           Show current migration version"
    echo "  reset             Drop all tables and re-run migrations"
    exit 1
    ;;
esac

---
# =====================================================
# Smoke Tests Script
# =====================================================
# scripts/smoke-tests.sh

#!/bin/bash

# Smoke tests for deployment validation

set -e

ENVIRONMENT=$1
BASE_URL=""

case $ENVIRONMENT in
  "staging")
    BASE_URL="https://staging-api.learningcompanion.ai"
    ;;
  "production")
    BASE_URL="https://api.learningcompanion.ai"
    ;;
  *)
    echo "Usage: ./smoke-tests.sh {staging|production}"
    exit 1
    ;;
esac

echo "Running smoke tests against $ENVIRONMENT ($BASE_URL)"

# Test 1: Health checks
echo "Testing health endpoints..."
for service in user-service ai-orchestrator study-buddy lab-service gamification content-service analytics; do
  response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health?service=$service")
  if [ "$response" != "200" ]; then
    echo "FAIL: $service health check failed (HTTP $response)"
    exit 1
  fi
  echo "  ✓ $service health check passed"
done

# Test 2: User registration
echo "Testing user registration..."
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test-'$(date +%s)'@example.com",
    "password": "TestPassword123",
    "display_name": "Smoke Test User",
    "role": "student"
  }')

http_code=$(echo "$response" | tail -n 1)
if [ "$http_code" != "201" ]; then
  echo "FAIL: User registration failed (HTTP $http_code)"
  exit 1
fi
echo "  ✓ User registration passed"

# Test 3: AI endpoint
echo "Testing AI endpoint..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/v1/ai/question" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?"}')

if [ "$response" != "200" ]; then
  echo "FAIL: AI endpoint failed (HTTP $response)"
  exit 1
fi
echo "  ✓ AI endpoint passed"

# Test 4: Database connectivity
echo "Testing database connectivity..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/progress/health-check")
if [ "$response" != "200" ]; then
  echo "FAIL: Database connectivity check failed (HTTP $response)"
  exit 1
fi
echo "  ✓ Database connectivity passed"

echo ""
echo "✅ All smoke tests passed for $ENVIRONMENT"
