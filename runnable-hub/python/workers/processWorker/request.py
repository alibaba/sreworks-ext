
from pydantic import BaseModel 

class Request(BaseModel):
    username: str
    email: str