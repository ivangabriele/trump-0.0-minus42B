.PHONY: download teach prepare generate tune train chat test type lint

download:
	python download.py
teach:
	@echo "Error: Please run this script manually: \`python teach.py <SAMPLE_SIZE>\`."
prepare:
	rm -fr models/rm && python prepare_reward_model.py
	rm -fr models/generator && python prepare_generator_model.py
generate:
	python generate.py
tune:
	python tune.py
train:
	@echo "Error: Not implemented yet.."
chat:
	python chat.py

test:
	.venv/bin/pytest -vv

type:
	.venv/bin/mypy .

lint:
	.venv/bin/ruff check .
