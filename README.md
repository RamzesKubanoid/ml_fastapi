# ml_fastapi
# All tests
pytest

# Unit tests only (faster)
pytest tests/ -k "not TestApi"

# One file
pytest tests/test_preprocessing.py -v

# Stop on first failure
pytest -x