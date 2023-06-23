from typing import Union
from src.api.schema_models.schema import InferenceRequest,InferenceResponse
from fastapi import FastAPI
from src.classifier.classifier import load_model,MyClassifier

app = FastAPI()

MODEL_PATH = ''
model: MyClassifier = load_model(MODEL_PATH)


@app.post("/news_classification",response_model=InferenceResponse)
async def inference_model(request:InferenceRequest):
    res = model.inference(text_list=request.text_list,get_prob=request.get_prob)
    return res

