.PHONY: chat data normalize scrap tune test

chat:
	.venv/bin/python ./chat.py
data:
	.venv/bin/python ./data.py
normalize:
	.venv/bin/python ./normalize.py
scrap:
	.venv/bin/python ./scrap.py
tune:
	.venv/bin/python ./tune.py

test:
	.venv/bin/pytest -vv
