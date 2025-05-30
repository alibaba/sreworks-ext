from pydantic import BaseModel

class LlmUsage(BaseModel):
    promptTokens: int
    completionTokens: int
    totalTokens: int