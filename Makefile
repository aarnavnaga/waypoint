SHELL := /bin/bash
GO ?= go
COMPOSE ?= docker compose -f deploy/compose/docker-compose.yml
PKGS := ./...
BIN := bin

GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
LDFLAGS := -X main.version=$(GIT_SHA)

.DEFAULT_GOAL := help

.PHONY: help
help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

## ---------- build ----------

.PHONY: build
build: ## Build gateway and mock backend into bin/
	@mkdir -p $(BIN)
	$(GO) build -ldflags "$(LDFLAGS)" -o $(BIN)/gateway ./cmd/gateway
	$(GO) build -ldflags "$(LDFLAGS)" -o $(BIN)/mockbackend ./cmd/mockbackend

.PHONY: tidy
tidy: ## Sync go.mod/go.sum
	$(GO) mod tidy

.PHONY: fmt
fmt: ## Format all Go code
	$(GO) fmt $(PKGS)

## ---------- test ----------

.PHONY: test
test: ## Run unit tests with the race detector
	$(GO) test -race -count=1 $(PKGS)

.PHONY: test-short
test-short: ## Run unit tests only (skip anything needing Docker/Redis)
	$(GO) test -short -count=1 $(PKGS)

.PHONY: cover
cover: ## Produce coverage report for internal/
	$(GO) test -race -count=1 -coverprofile=coverage.out -covermode=atomic ./internal/...
	$(GO) tool cover -func=coverage.out | tail -1
	$(GO) tool cover -html=coverage.out -o coverage.html

.PHONY: integration
integration: ## Run integration tests (needs Redis; Toxiproxy tests skip without it)
	$(GO) test -race -count=1 -tags=integration ./test/integration/...

.PHONY: deps-up
deps-up: ## Start just the dependencies integration tests and benchmarks need
	docker run -d --name waypoint-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v1 || true
	docker run -d --name waypoint-toxiproxy -p 8474:8474 -p 19100-19110:19100-19110 ghcr.io/shopify/toxiproxy:2.9.0 || true

.PHONY: deps-down
deps-down: ## Stop the standalone dependencies
	docker rm -f waypoint-redis waypoint-toxiproxy 2>/dev/null || true

.PHONY: fuzz
fuzz: ## Run fuzz targets briefly
	$(GO) test -run=XXX -fuzz=FuzzParseChatRequest -fuzztime=30s ./internal/openai
	$(GO) test -run=XXX -fuzz=FuzzSSEDecoder -fuzztime=30s ./internal/sse

.PHONY: lint
lint: ## Run golangci-lint (installs it if absent)
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo "golangci-lint not found; install it with: brew install golangci-lint"; \
		exit 1; \
	}
	golangci-lint run --timeout 5m

.PHONY: vet
vet: ## Run go vet
	$(GO) vet $(PKGS)

.PHONY: check
check: fmt vet test ## Format, vet and test

## ---------- local stack ----------

.PHONY: up
up: ## Start the full local stack
	$(COMPOSE) up -d --build
	@echo "gateway   http://localhost:8080"
	@echo "grafana   http://localhost:3000 (admin/admin)"
	@echo "jaeger    http://localhost:16686"
	@echo "prom      http://localhost:9090"

.PHONY: down
down: ## Stop the stack and remove volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail stack logs
	$(COMPOSE) logs -f --tail=100

.PHONY: ps
ps: ## Show stack status
	$(COMPOSE) ps

.PHONY: smoke
smoke: ## Send one streaming request through the gateway
	curl -sN http://localhost:8080/v1/chat/completions \
		-H 'Content-Type: application/json' \
		-H 'Authorization: Bearer dev-key' \
		-d '{"model":"mock-7b","stream":true,"messages":[{"role":"user","content":"hello"}]}'

## ---------- benchmarks ----------

.PHONY: bench-routing
bench-routing: ## Routing policy comparison (M8)
	bench/run.sh routing

.PHONY: bench-caching
bench-caching: ## Cache hit rate / latency / quality tradeoff (M8)
	bench/run.sh caching

.PHONY: bench-resilience
bench-resilience: ## Fault injection via Toxiproxy (M8)
	bench/run.sh resilience

.PHONY: bench-scaling
bench-scaling: ## Horizontal scaling of gateway replicas (M8)
	bench/run.sh scaling

.PHONY: bench-report
bench-report: ## Turn bench/results/*.json into tables and charts
	@test -x bench/.venv/bin/python || { \
		echo "creating bench venv..."; \
		python3 -m venv bench/.venv && bench/.venv/bin/pip install --quiet matplotlib; \
	}
	bench/.venv/bin/python bench/report.py

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf $(BIN) coverage.out coverage.html
