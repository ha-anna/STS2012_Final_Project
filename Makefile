VENV_PATH=.venv

# Create venv and install dependencies
setup:
	poetry config virtualenvs.in-project true
	poetry install

# Save current environment to poetry.lock (auto-managed)
freeze:
	poetry lock

# Install from pyproject.toml and poetry.lock
install:
	poetry install

# Add a new dependency via poetry and update lock file
add-dep:
	@echo "Usage: make add-dep pkg=package_name"
	@if [ -z "$(pkg)" ]; then echo "Error: pkg variable not set"; exit 1; fi
	poetry add $(pkg)

# Remove a dependency using Poetry
remove-dep:
	@poetry remove $(pkg)

# Train the CNN model
train:
	poetry run python -m src.cnn.cnn

# Run the ASL recognition script
run:
	poetry run python -m src.sign_recognition.sign_recognition

# Format and sort imports
lint:
	poetry run black src
	poetry run isort src

# Run intro script
intro:
	poetry run python -m src.main

# Show stats
stats:
	poetry run python -m src.cnn.stats
