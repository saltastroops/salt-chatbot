from typing import Optional, List
from pydantic import BaseModel

class QueryInput(BaseModel):
    text: str

class QueryOutput(BaseModel):
    answer: str
    sources: List[str]
    intermediate_steps: Optional[List[str]]