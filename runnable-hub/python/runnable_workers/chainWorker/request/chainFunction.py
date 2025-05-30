from json import tool
from pydantic import BaseModel
from typing import Dict, Optional, List
from enum import Enum
from runnable_hub import RunnableValueDefine

class ChainFunctionType(str, Enum):
    TOOL = "TOOL"
    AGENT = "AGENT"

class ChainFunction(BaseModel):
    type: ChainFunctionType
    name: str
    version: str
    inputDefine: List[RunnableValueDefine] = []
    outputDefine: List[RunnableValueDefine] = []
    presetInputs: Dict = {}
    description: Optional[str] = None

