# Job Hunt OS — Docker convenience commands.
# `make up` is the one-liner: creates .env on first run, builds, starts everything,
# waits for Postgres, and makes sure the litellm database exists.

.DEFAULT_GOAL := help

.PHONY: help up down restart logs ps env wait-db litellm-db backup db-shell shell clean

help:
	@echo "Job Hunt OS — docker compose shortcuts"
	@echo ""
	@echo "  make up          Build and start everything (first run also creates .env + litellm DB)"
	@echo "  make down        Stop all containers"
	@echo "  make restart     Rebuild and restart just the web app (after a code change)"
	@echo "  make logs        Tail logs from all containers"
	@echo "  make ps          Show container status"
	@echo "  make backup      Dump the jobhunt database into backups/"
	@echo "  make db-shell    Open a psql shell on the jobhunt database"
	@echo "  make shell       Open a shell inside the web container"
	@echo "  make clean       Stop containers AND delete all data — destructive"

env:
	@test -f .env || (cp .env.example .env && \
		echo "Created .env from .env.example — add your API keys (SECRET_KEY, APIFY_API_TOKEN, an LLM key or DeepSeek+LiteLLM) before running again.")

up: env
	docker compose up --build -d
	@$(MAKE) --no-print-directory wait-db
	@$(MAKE) --no-print-directory litellm-db
	@echo ""
	@echo "Job Hunt OS is running: http://localhost:8420"
	@echo "LiteLLM proxy (if configured):  http://localhost:4000"
	@echo "'make logs' to watch it, 'make down' to stop."

down:
	docker compose down

restart:
	docker compose up -d --build web

logs:
	docker compose logs -f

ps:
	docker compose ps

wait-db:
	@echo "Waiting for the database..."
	@for i in $$(seq 1 30); do \
		docker compose exec -T db pg_isready -U jobhunt >/dev/null 2>&1 && exit 0; \
		sleep 1; \
	done; \
	echo "Database did not become ready in time" >&2

litellm-db:
	@docker compose exec -T db psql -U jobhunt -d jobhunt -tc \
		"SELECT 1 FROM pg_database WHERE datname='litellm'" 2>/dev/null | grep -q 1 || \
	docker compose exec -T db psql -U jobhunt -d jobhunt -c "CREATE DATABASE litellm OWNER jobhunt;" >/dev/null 2>&1 || true

backup:
	@mkdir -p backups
	docker compose exec -T db pg_dump -U jobhunt -d jobhunt -F c > backups/jobhunt_$$(date +%Y%m%d_%H%M%S).dump
	@echo "Backup saved under backups/"

db-shell:
	docker compose exec db psql -U jobhunt -d jobhunt

shell:
	docker compose exec web sh

clean:
	docker compose down -v
