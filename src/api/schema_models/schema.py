
from typing import List,Dict,Any
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    text_list: List[str]
    get_prob: bool = True

class InferenceResponse(BaseModel):
    result: List[Dict[str,Any]]
