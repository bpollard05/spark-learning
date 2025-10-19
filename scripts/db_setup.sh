# =====================================================
# FILE: scripts/db_setup.sh
# LOCATION: scripts/db_setup.sh
# Convenience script for database operations
# =====================================================

#!/bin/bash

# Database setup and migration script

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🗄️  Database Setup Script${NC}"
echo ""

# Check if alembic is installed
if ! command -v alembic &> /dev/null; then
    echo -e "${RED}❌ Alembic not found. Installing...${NC}"
    pip install alembic
fi

COMMAND=$1

case $COMMAND in
    "init")
        echo -e "${YELLOW}📦 Initializing Alembic...${NC}"
        alembic init migrations
        echo -e "${GREEN}✅ Alembic initialized!${NC}"
        echo "Don't forget to:"
        echo "  1. Update migrations/env.py with your models"
        echo "  2. Update alembic.ini with your database URL"
        ;;
    
    "create")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Migration message required${NC}"
            echo "Usage: ./db_setup.sh create 'migration message'"
            exit 1
        fi
        echo -e "${YELLOW}📝 Creating new migration: $2${NC}"
        alembic revision --autogenerate -m "$2"
        echo -e "${GREEN}✅ Migration created!${NC}"
        ;;
    
    "upgrade"|"up")
        echo -e "${YELLOW}⬆️  Running migrations...${NC}"
        alembic upgrade head
        echo -e "${GREEN}✅ Database updated!${NC}"
        ;;
    
    "downgrade"|"down")
        echo -e "${YELLOW}⬇️  Rolling back last migration...${NC}"
        alembic downgrade -1
        echo -e "${GREEN}✅ Rolled back!${NC}"
        ;;
    
    "current")
        echo -e "${YELLOW}📍 Current migration version:${NC}"
        alembic current
        ;;
    
    "history")
        echo -e "${YELLOW}📜 Migration history:${NC}"
        alembic history --verbose
        ;;
    
    "reset")
        echo -e "${RED}⚠️  WARNING: This will drop all tables and re-run migrations${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo -e "${YELLOW}🗑️  Dropping all tables...${NC}"
            alembic downgrade base
            echo -e "${YELLOW}⬆️  Re-running migrations...${NC}"
            alembic upgrade head
            echo -e "${GREEN}✅ Database reset complete!${NC}"
        else
            echo -e "${YELLOW}Cancelled.${NC}"
        fi
        ;;
    
    "seed")
        echo -e "${YELLOW}🌱 Seeding database...${NC}"
        python scripts/seed_database.py
        echo -e "${GREEN}✅ Database seeded!${NC}"
        ;;
    
    *)
        echo "Usage: ./db_setup.sh {command}"
        echo ""
        echo "Commands:"
        echo -e "  ${GREEN}init${NC}              Initialize Alembic (first time only)"
        echo -e "  ${GREEN}create 'message'${NC}  Create a new migration"
        echo -e "  ${GREEN}up${NC}                Run all pending migrations"
        echo -e "  ${GREEN}down${NC}              Rollback last migration"
        echo -e "  ${GREEN}current${NC}           Show current migration version"
        echo -e "  ${GREEN}history${NC}           Show migration history"
        echo -e "  ${GREEN}reset${NC}             Drop all tables and re-run migrations"
        echo -e "  ${GREEN}seed${NC}              Seed database with test data"
        echo ""
        exit 1
        ;;
esac