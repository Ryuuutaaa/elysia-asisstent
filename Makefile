.PHONY: install install-dev lint test ci

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest -o addopts="" -q

ci: lint test
