from pydantic import BaseModel

class LlmSetting(BaseModel):
    endpoint: str
    model: str
    secretKey: str