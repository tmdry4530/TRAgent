.PHONY: dev test lint backtest deploy

dev:
	python -m src.main

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/
	mypy src/

backtest:
	python -m src.backtest.run

backtest-scalp:
	python -m src.backtest.run --strategy scalp

backtest-swing:
	python -m src.backtest.run --strategy swing

deploy:
	./scripts/deploy.sh
