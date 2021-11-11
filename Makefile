.PHONY: init
init:
	poetry install --extras "torch"
	pre-commit install

.PHONY: format
format:
	pre-commit run -a
