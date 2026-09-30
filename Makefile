# Everything runs in Docker: the host needs only docker compose (no Python, no uv).
.PHONY: up down logs health ingest eval eval-translate test lint fmt

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

test:    ## ruff + pytest, offline and without LLM keys, on the reformulation_test database
	docker compose run --rm test

lint:    ## ruff only
	docker compose run --rm test sh -c "ruff check . && ruff format --check ."

fmt:     ## apply ruff formatting and safe fixes
	docker compose run --rm test sh -c "ruff check --fix . && ruff format ."
