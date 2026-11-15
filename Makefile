# 5a-tender-checker: one command per job.   make help
PY       = uv run python -m tender
# LLM_BUDGET_USD=0.10 make check ...  -> calls stop once traced spend reaches $0.10 (per command)
PROJECT ?=
FILES   ?=
FORMAT  ?= md
FILTER  ?=
SCOPE   ?= both
GOLD    ?=
Q       ?= What documents must be submitted with the bid?
PORT    ?= 8000
NAME    ?=

.PHONY: demo-sets serve user docker-up docker-user docker-down docker-logs help setup smoke models samples new list ingest-tender ingest-bid check matrix ask delete \
        demo demo-redteam synth eval redteam retrieval-eval trace test

help:
	@echo "Setup"
	@echo "  make setup                         uv sync + .env + prices.toml"
	@echo "  make smoke [TIER=pro]              one LLM call + one embedding call (Gemini by default)"
	@echo "  make models [FILTER=flash]         models your Gemini key can use (live list, with prices)"
	@echo "Service (API + web UI)"
	@echo "  make user NAME=sajid               create an admin user, prints the API key (once)"
	@echo "  make serve                         http://127.0.0.1:8000  (UI at /, API docs at /docs)"
	@echo "  make docker-up                     same in Docker: API + Qdrant + Postgres (needs Docker)"
	@echo "  make docker-user NAME=sajid        create a user inside the running containers"
	@echo "  make docker-down | docker-logs"
	@echo "Use (CLI)"
	@echo "  make new NAME=\"...\"                create a project (prints its id)"
	@echo "  make ingest-tender PROJECT=p-.. FILES=\"tender.pdf\""
	@echo "  make ingest-bid PROJECT=p-.. FILES=\"bid.docx annex.pdf\""
	@echo "  make check PROJECT=p-..            requirement extraction -> judge -> review -> gap summary"
	@echo "  make matrix PROJECT=p-.. [FORMAT=md|json|csv|xlsx]"
	@echo "  make ask PROJECT=p-.. Q=\"...\" [SCOPE=tender|bid|both]"
	@echo "  make delete PROJECT=p-..           files, chunks, vectors, results"
	@echo "  make list"
	@echo "Demo + evaluation"
	@echo "  make samples                       fictional tender PDF + bid DOCX with planted gaps + gold key"
	@echo "  make demo-sets                     2 real small RFQs x 3 bids (strong / weak / poisoned) + answer keys"
	@echo "  make demo                          samples -> project 'demo' -> check -> eval vs gold"
	@echo "  make demo-redteam                  poisoned bid vs the clean demo -> verdicts changed (target 0)"
	@echo "  make synth PROJECT=p-..            synthetic bid with planted gaps for the project's tender"
	@echo "  make eval PROJECT=p-.. GOLD=path   score the latest check"
	@echo "  make retrieval-eval PROJECT=demo GOLD=samples/gold_acme.json   recall@k: dense vs keyword vs hybrid"
	@echo "  make trace RUN=<run_id>            tokens / cost / latency of a run"
	@echo "  make test                          offline tests (fake LLM, in-memory Qdrant)"

setup:
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env: fill in GEMINI_API_KEY")
	@test -f prices.toml || cp prices.example.toml prices.toml

smoke:
	uv run python -m agentcore.smoke $(TIER)

models:
	uv run python -m agentcore.models $(FILTER)

samples:
	$(PY) samples

demo-sets:
	$(PY) demo-sets

new:
	$(PY) new "$(NAME)"

list:
	$(PY) list

ingest-tender:
	$(PY) ingest $(PROJECT) $(FILES) --role tender

ingest-bid:
	$(PY) ingest $(PROJECT) $(FILES) --role bid

check:
	$(PY) check $(PROJECT)

matrix:
	$(PY) matrix $(PROJECT) --format $(FORMAT)

ask:
	$(PY) ask $(PROJECT) "$(Q)" --scope $(SCOPE)

delete:
	$(PY) delete $(PROJECT) --yes

demo: samples
	@$(PY) delete demo --yes >/dev/null 2>&1 || true
	$(PY) new "Demo: cloud hosting RFP (fictional)" --id demo
	$(PY) ingest demo samples/tender_cloud_hosting.pdf --role tender
	$(PY) ingest demo samples/bid_acme.docx --role bid
	$(PY) check demo
	$(PY) eval demo --gold samples/gold_acme.json

# reuses the clean 'demo' check if one exists (no second paid run); runs `make demo` only if not
demo-redteam: samples
	@$(PY) matrix demo --format json --out /dev/null >/dev/null 2>&1 || $(MAKE) demo
	@$(PY) delete demo-poisoned --yes >/dev/null 2>&1 || true
	$(PY) new "Demo: poisoned bid" --id demo-poisoned
	$(PY) ingest demo-poisoned samples/tender_cloud_hosting.pdf --role tender
	$(PY) ingest demo-poisoned samples/bid_acme_poisoned.docx --role bid
	$(PY) check demo-poisoned
	$(PY) redteam demo demo-poisoned --gold samples/gold_acme.json

synth:
	$(PY) synth $(PROJECT)

eval:
	$(PY) eval $(PROJECT) --gold $(GOLD)

retrieval-eval:
	$(PY) retrieval-eval $(PROJECT) --gold $(GOLD)

trace:
	$(PY) trace $(RUN)

test:
	uv run pytest -q

# ---------------------------------------------------------------- service
# Embedded Qdrant allows ONE process: stop `make serve` before running CLI commands that touch vectors
# (ingest / check / ask / retrieval-eval), or use docker compose (Qdrant server, no such limit).
serve:
	uv run tender serve --port $(PORT)

user:
	uv run tender user create $(NAME) --admin

docker-up:
	@test -f .env || (echo "Create .env first: make setup, then add GEMINI_API_KEY and POSTGRES_PASSWORD"; exit 1)
	docker compose up -d --build
	@echo "UI: http://localhost:8000   API docs: http://localhost:8000/docs   health: http://localhost:8000/health"

docker-user:
	docker compose exec api tender user create $(NAME) --admin

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api
