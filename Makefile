.PHONY: download teach prepare generate tune train chat test

download:
	.venv/bin/python ./download.py
teach:
	@echo "Error: Please run this script manually: \`python ./teach.py <SAMPLE_SIZE>\`."
prepare:
	@echo "Error: Not implemented yet.."
generate:
	.venv/bin/python ./generate.py
tune:
	.venv/bin/python ./tune.py
train:
	@echo "Error: Not implemented yet.."
chat:
	.venv/bin/python ./chat.py

test:
	.venv/bin/pytest -vv
