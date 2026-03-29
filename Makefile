.PHONY: evaluate

RUN ?=
CFG ?=
GEN ?=
DATASET ?=
MAX_ITEMS ?=
COOLDOWN ?=
THINKING ?=
RESUME ?=
RUN_CONFIG = configs/runs/$(RUN)/$(CFG)
EVAL_ARGS = --run-config "$(RUN_CONFIG)" $(if $(GEN),--generation-path "$(GEN)") $(if $(DATASET),--dataset-path "$(DATASET)") $(if $(MAX_ITEMS),--max-items "$(MAX_ITEMS)") $(if $(RESUME),--resume)

evaluate:
	@test -n "$(RUN)" || (echo "Usage: make evaluate RUN=<run-folder> CFG=<yaml-file> [GEN=<jsonl>] [DATASET=<path>] [MAX_ITEMS=<n>] [COOLDOWN=<seconds>] [THINKING=<level>]" && exit 1)
	@test -n "$(CFG)" || (echo "Usage: make evaluate RUN=<run-folder> CFG=<yaml-file> [GEN=<jsonl>] [DATASET=<path>] [MAX_ITEMS=<n>] [COOLDOWN=<seconds>] [THINKING=<level>]" && exit 1)
	PYTHONPATH=src \
	EXGEN_GEMINI_COOLDOWN_SECONDS="$(if $(COOLDOWN),$(COOLDOWN),$${EXGEN_GEMINI_COOLDOWN_SECONDS:-0.05})" \
	EXGEN_GEMINI_THINKING_LEVEL="$(if $(THINKING),$(THINKING),$${EXGEN_GEMINI_THINKING_LEVEL:-low})" \
	uv run --active --no-project python -m exgen.pipeline.run_evaluation $(EVAL_ARGS)
