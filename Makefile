.PHONY: install test smoke main ablation lint format clean

install:
	pip install -e ".[dev,log]"

test:
	pytest -q tests

smoke:
	python -m lsa_node.train --config configs/smoke.yaml

main:
	bash scripts/reproduce_table_main.sh

ablation:
	bash scripts/reproduce_table_ablation.sh

lint:
	ruff check lsa_node tests

format:
	ruff format lsa_node tests

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ build dist *.egg-info
