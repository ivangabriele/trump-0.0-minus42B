.PHONY: chat data scrap tune

chat:
	.venv/bin/python ./chat.py
data:
	.venv/bin/python ./data.py
scrap:
	.venv/bin/python ./scrap.py
tune:
	.venv/bin/python ./tune.py
