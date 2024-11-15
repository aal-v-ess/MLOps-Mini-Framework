install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=pipeline tests/test_*.py

ruff:
	ruff check . --fix

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: 
	format lint ruff

deploy:
	#deploy goes here
		
all: install ruff format test deploy