
from pydantic import BaseModel

class LlmMessage(BaseModel):
    role: str
    content: str
