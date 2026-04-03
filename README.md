# ml_fastapi
# All tests
pytest

# Unit tests only (faster)
pytest tests/ -k "not TestApi"

# One file
pytest tests/test_preprocessing.py -v

# Stop on first failure
pytest -x


# Build
docker build -t churn-service .

# Run — mount models/ as a volume so trained models survive restarts
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-service

# Verify endpoints inside the container
curl http://localhost:8000/health
curl http://localhost:8000/docs        # Swagger UI
curl http://localhost:8000/model/schema