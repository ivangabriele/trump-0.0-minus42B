.PHONY: chat data normalize prepare tune test

download:
	.venv/bin/python ./download.py
prepare:
	.venv/bin/python ./prepare.py
generate:
	.venv/bin/python ./generate.py
tune:
	.venv/bin/python ./tune.py
chat:
	.venv/bin/python ./chat.py

test:
	.venv/bin/pytest -vv
