.PHONY: chat data scrap train

chat:
	.venv/bin/python ./chat.py
data:
	.venv/bin/python ./data.py
scrap:
	.venv/bin/python ./scrap.py
train:
	.venv/bin/python ./train.py
