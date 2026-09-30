# Everything runs in Docker: the host needs only docker compose (no Python, no uv).
.PHONY: up down logs health ingest eval eval-translate tokens live-runs test lint fmt \
	k8s-up k8s-ingest k8s-status k8s-forward k8s-down

# --- Kubernetes on kind (k8s/) -------------------------------------------------------------
# kind installed without Homebrew lives in ~/bin, which may not be on make's PATH.
KIND ?= $(shell command -v kind 2>/dev/null || echo $(HOME)/bin/kind)
KIND_CLUSTER = pet
# Every kubectl call names the kind context: never Docker Desktop's own cluster by accident.
KUBECTL = kubectl --context kind-$(KIND_CLUSTER) -n pet
# The tags come from the manifests, so they live in one place.
API_IMAGE := $(shell awk '/image: pet-api/ {print $$2; exit}' k8s/api-deployment.yaml)
PG_IMAGE := $(shell awk '/image: pgvector/ {print $$2; exit}' k8s/postgres-statefulset.yaml)
# The kind node's platform (linux/arm64 on Apple silicon). `kind load docker-image` imports all
# platforms of a pulled multi-arch image (pgvector) and fails on the layers Docker never
# downloaded ("content digest ... not found"); saving only this platform avoids that.
NODE_PLATFORM := linux/$(shell docker version -f '{{.Server.Arch}}' 2>/dev/null)

up:      ## build and start db + api in the background
	docker compose up --build -d

down:    ## stop the stack (data in the pgdata volume is kept)
	docker compose down

logs:    ## follow the api JSON logs
	docker compose logs -f api

health:  ## check that the api and the database answer
	curl -s localhost:8000/health

ingest:  ## load data/corpus/ into the knowledge base (re-ingest replaces, never duplicates)
	docker compose exec api python -m app.ingest data/corpus/

eval:    ## recall@5 of /ask search, en vs uk questions (search only, no LLM calls)
	docker compose exec api python -m eval.recall

eval-translate:  ## same + uk questions translated as /ask does (live LLM: 1 call per question)
	docker compose exec api python -m eval.recall --translate

tokens:  ## token usage of recent /reformulate runs (make tokens ARGS="--last 3")
	docker compose exec api python -m eval.tokens $(ARGS)

live-runs:  ## live /reformulate runs, 3 goals x 5, checked (LIVE LLM: ~30-60 calls, ~10 min)
	docker compose run --rm live python scripts/live_runs.py $(ARGS)

test:    ## ruff + pytest, offline and without LLM keys, on the reformulation_test database
	docker compose run --rm test

lint:    ## ruff only
	docker compose run --rm test sh -c "ruff check . && ruff format --check ."

fmt:     ## apply ruff formatting and safe fixes
	docker compose run --rm test sh -c "ruff check --fix . && ruff format ."

k8s-up:  ## kind cluster + image + Secrets + manifests, waits until postgres and api are ready
	$(KIND) get clusters | grep -qx $(KIND_CLUSTER) || $(KIND) create cluster --name $(KIND_CLUSTER)
	docker build -t $(API_IMAGE) .
	docker image inspect $(PG_IMAGE) >/dev/null 2>&1 || docker pull $(PG_IMAGE)
	@for image in $(API_IMAGE) $(PG_IMAGE); do \
		archive=$$(mktemp) && echo "kind load $$image ($(NODE_PLATFORM))" && \
		docker save --platform $(NODE_PLATFORM) $$image -o $$archive && \
		$(KIND) load image-archive $$archive --name $(KIND_CLUSTER); \
		status=$$?; rm -f $$archive; [ $$status -eq 0 ] || exit $$status; \
	done
	$(KUBECTL) apply -f k8s/namespace.yaml
	@# LLM keys and settings from .env; apply keeps a re-run from failing on "already exists".
	$(KUBECTL) create secret generic pet-llm --from-env-file=.env --dry-run=client -o yaml \
		| $(KUBECTL) apply -f -
	@# The Postgres password is made once: a new one would not match the database on the PVC.
	@# Hex only: it goes straight into DATABASE_URL, where @ / : would break the URL.
	$(KUBECTL) get secret pet-postgres >/dev/null 2>&1 || $(KUBECTL) create secret generic \
		pet-postgres --from-literal=password=$$(openssl rand -hex 16)
	$(KUBECTL) apply -k k8s/
	@# Same tag after a rebuild: the pod would keep the old image without a restart.
	$(KUBECTL) rollout restart deployment/api
	$(KUBECTL) rollout status statefulset/postgres --timeout=180s
	$(KUBECTL) rollout status deployment/api --timeout=300s

k8s-ingest:  ## load the corpus baked into the image: re-runs the ingest Job, prints its log
	$(KUBECTL) delete job ingest --ignore-not-found
	$(KUBECTL) apply -f k8s/ingest-job.yaml
	@# Complete or Failed, whichever comes first (kubectl wait knows one condition only).
	@for i in $$(seq 1 150); do \
		state=$$($(KUBECTL) get job ingest \
			-o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}'); \
		case "$$state" in \
			*Complete=True*) break ;; \
			*Failed=True*) $(KUBECTL) logs job/ingest -c ingest --tail=30; exit 1 ;; \
		esac; \
		sleep 2; \
	done; \
	case "$$state" in *Complete=True*) ;; *) echo "ingest: no result after 300 s"; exit 1 ;; esac
	$(KUBECTL) logs job/ingest -c ingest | grep -v '^{'

k8s-status:  ## pods, services, volumes and the latest events in the pet namespace
	$(KUBECTL) get pods,svc,pvc -o wide
	$(KUBECTL) get events --sort-by=.lastTimestamp | tail -n 15

k8s-forward:  ## the api on localhost:8001 (8000 is taken by compose); Ctrl+C to stop
	$(KUBECTL) port-forward svc/api 8001:8000

k8s-down:  ## delete the kind cluster (its volumes too)
	$(KIND) delete cluster --name $(KIND_CLUSTER)
