
from .model.bert_classifier import BertClassifierModel, get_model
from pydantic import BaseModel, ValidationError, validator
from fastapi import FastAPI
from fastapi import Depends
from typing import List
import logging


logger = logging.getLogger('api')
def setup_logging():
    screen_formatter = logging.Formatter('[%(levelname)s] - %(message)s')
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(screen_formatter)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler('api.log', maxBytes=100000, backupCount=10)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(screen_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


class ClassifyRequest(BaseModel):
    data: List[str]
    
    @validator("data")
    def check_length(cls, v):
        max_input_length = 100
        if not v or len(v) >= max_input_length:
            raise ValidationError(f'0 < valid input length <= {max_input_length}')
        return v

class ClassifyResponse(BaseModel):
    data: List[int]


app = FastAPI()
ml_model = None
@app.on_event("startup")
async def startup_event():
    setup_logging()
    logger.info('setup complete')

@app.post("/classify", response_model=ClassifyResponse)
def classify(input: ClassifyRequest, model: BertClassifierModel = Depends(get_model)):
    logger.info(f'starting model prediction.')
    logger.debug(f'input={input.data}')
    pred = model.predict(input.data)
    logger.debug(f'output={pred}')
    return ClassifyResponse(data=pred)