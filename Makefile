.PHONY: download teach prepare generate tune train chat test type lint

download:
	python download.py
teach:
	@echo "Error: Please run this script manually: \`python teach.py <SAMPLE_SIZE>\`."
prepare:
	python prepare_reward_model.py
	python prepare_normalizer_model.py
normalize:
	python normalize.py
tune:
	python tune.py
train:
	@echo "Error: Not implemented yet.."
chat:
	python chat.py

serve:
	# `7860` is the default port for Hugging Face Spaces running on Docker
	# https://huggingface.co/docs/hub/en/spaces-config-reference
	@echo "Info: Starting HTTP server on port 7860…"
	python -m http.server --directory public 7860

test:
	.venv/bin/pytest -vv

type:
	.venv/bin/mypy .

lint:
	.venv/bin/ruff check .
