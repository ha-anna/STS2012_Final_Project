setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	python3 -m venv .venv
	source .venv/bin/activate

freeze:
	python3 -m pip freeze > requirements.txt

install:
	python3 -m pip install -r requirements.txt

train:
	python -m src.cnn.cnn

run:
	python -m src.sign_recognition.sign_recognition

lint:
	black src
	isort src