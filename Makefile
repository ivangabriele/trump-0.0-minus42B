.PHONY: chat data normalize prepare tune test

chat:
	.venv/bin/python ./chat.py
download:
	.venv/bin/python ./download.py
prepare:
	.venv/bin/python ./prepare.py
normalize:
	.venv/bin/python ./normalize.py
tune:
	.venv/bin/python ./tune.py

test:
	.venv/bin/pytest -vv
