# Makefile

VENV_PATH=.venv
ACTIVATE=source $(VENV_PATH)/bin/activate

# Create venv and install dependencies
setup:
	python3 -m pip install --upgrade pip
	python3 -m venv $(VENV_PATH)
	$(ACTIVATE) && pip install -r requirements.txt

# Save current environment to requirements.txt
freeze:
	$(ACTIVATE) && pip freeze > requirements.txt

# Install from requirements.txt
install:
	$(ACTIVATE) && pip install -r requirements.txt

# Train the CNN model
train:
	$(ACTIVATE) && python -m src.cnn.cnn

# Run the ASL recognition script
run:
	$(ACTIVATE) && python -m src.sign_recognition.sign_recognition

# Format and sort imports
lint:
	black src
	isort src

# Run intro script
intro:
	$(ACTIVATE) && python -m src.main

# Show stats
stats:
	$(ACTIVATE) && python -m src.cnn.stats
