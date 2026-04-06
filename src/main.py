"""ML churn service — application entry point."""
from fastapi import FastAPI

from src.error_handlers import register_error_handlers
from src.api.health import router as health_router
from src.api.api_dataset import router as dataset_router
from src.api.api_model import router as model_router
from src.api.predict import router as predict_router

app = FastAPI(
    title="Churn Prediction Service",
    description=(
        "Trains machine learning models to predict customer churn "
        "and serves real-time predictions via a REST API."
    ),
    version="1.0.0",
)

register_error_handlers(app)


@app.get("/", tags=["Root"])
def root():
    """Service root — confirms the API is reachable."""
    return {"message": "ml churn service is running"}


app.include_router(health_router)
app.include_router(dataset_router, prefix="/dataset", tags=["Dataset"])
app.include_router(model_router,   prefix="/model",   tags=["Model"])
app.include_router(predict_router,                    tags=["Prediction"])
