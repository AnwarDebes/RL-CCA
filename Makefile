# Makefile - shortcuts for common research-subproject tasks.
# Use `make <target>` from the nexus root.

VENV := ./venv/bin/python
PYTEST := $(VENV) -m pytest --tb=short

.PHONY: help test test-fast test-flagship test-cmaz test-wreath \
        smoke summary compare benchmarks docs clean install-deps \
        env validate verify-phase1 verify-phase2 status

help:
	@echo "Targets:"
	@echo "  make test           - run all unit tests across 3 subprojects (~3 min)"
	@echo "  make test-fast      - run only math-only tests (~30s)"
	@echo "  make test-flagship  - run flagship subproject tests"
	@echo "  make test-cmaz      - run CMAZ tests"
	@echo "  make test-wreath    - run wreath equivariant tests"
	@echo "  make smoke          - run full integration demo (~30s)"
	@echo "  make summary        - print architecture summary"
	@echo "  make compare        - print subproject comparison table"
	@echo "  make docs           - regenerate auto-generated docs (architecture.tex)"
	@echo "  make env            - verify Python environment + GPU"
	@echo "  make validate       - comprehensive repo health check (~3 min)"
	@echo "  make status         - quick status snapshot (training + project + GPU)"
	@echo "  make verify-phase1  - sanity-check Phase 1 v4 RL training"
	@echo "  make verify-phase2  - sanity-check Phase 2 v4 RL training"
	@echo "  make launch-phase2  - verify Phase 1 + launch Phase 2"
	@echo "  make phase2-eta     - predict when Phase 2 will finish"
	@echo "  make restart-phase2 - resume Phase 2 from latest checkpoint after crash"
	@echo "  make dashboard      - live status (training + GPU + checkpoints)"
	@echo "  make benchmarks     - show available scripts (BENCHMARKS.md)"
	@echo "  make clean          - remove __pycache__ and .pytest_cache"

test:
	./run_all_tests.sh

test-fast:
	$(PYTEST) \
	  flagship_coalition_mcts/tests/test_plackett_luce.py \
	  flagship_coalition_mcts/tests/test_coalition_head.py \
	  flagship_coalition_mcts/tests/test_cce_selector.py \
	  flagship_coalition_mcts/tests/test_kingmaker.py \
	  flagship_coalition_mcts/tests/test_exploitability.py \
	  flagship_coalition_mcts/tests/test_halma_small.py \
	  flagship_coalition_mcts/tests/test_head_to_head.py \
	  flagship_coalition_mcts/tests/test_results_table.py \
	  flagship_coalition_mcts/tests/test_replay_buffer.py \
	  flagship_coalition_mcts/tests/test_subtree_reuse.py
	$(PYTEST) decomposed_mcts/tests/test_monotonic_mixer.py \
	          decomposed_mcts/tests/test_cmaz_mcts.py
	$(PYTEST) equivariant_net/tests/test_seat_equivariant.py \
	          equivariant_net/tests/test_c6_spatial.py \
	          equivariant_net/tests/test_wreath_fuse.py

test-flagship:
	$(PYTEST) flagship_coalition_mcts/tests/

test-cmaz:
	$(PYTEST) decomposed_mcts/tests/

test-wreath:
	$(PYTEST) \
	  equivariant_net/tests/test_seat_equivariant.py \
	  equivariant_net/tests/test_c6_spatial.py \
	  equivariant_net/tests/test_wreath_fuse.py \
	  equivariant_net/tests/test_cc_wreath_encoder.py \
	  equivariant_net/tests/test_wreath_network.py \
	  equivariant_net/tests/test_cc_runner.py \
	  equivariant_net/tests/test_public_api.py

smoke:
	$(VENV) flagship_coalition_mcts/experiments/full_integration_demo.py \
	  --num-players 2 --num-simulations 4

summary:
	$(VENV) -m flagship_coalition_mcts.src.model_summary

compare:
	$(VENV) -m flagship_coalition_mcts.src.compare_subprojects

docs:
	$(VENV) -m flagship_coalition_mcts.src.model_summary --format latex \
	  > flagship_coalition_mcts/docs/architecture.tex
	$(VENV) -m flagship_coalition_mcts.src.compare_subprojects --format latex \
	  > flagship_coalition_mcts/docs/subproject_comparison.tex
	@echo "Regenerated docs/architecture.tex, docs/subproject_comparison.tex"

benchmarks:
	@cat BENCHMARKS.md | head -50

clean:
	find flagship_coalition_mcts decomposed_mcts equivariant_net \
	  -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find flagship_coalition_mcts decomposed_mcts equivariant_net \
	  -name .pytest_cache -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned __pycache__ and .pytest_cache"

env:
	$(VENV) check_env.py

validate:
	./validate_repo.sh

status:
	./scripts/quick_status.sh

verify-phase1:
	$(VENV) scripts/verify_phase1_complete.py

verify-phase2:
	$(VENV) scripts/verify_phase2_complete.py

launch-phase2:
	./scripts/launch_phase2.sh

phase2-eta:
	$(VENV) scripts/phase2_eta.py

restart-phase2:
	./scripts/restart_phase2.sh

dashboard:
	./scripts/dashboard.sh
