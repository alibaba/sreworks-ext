from json import tool
from pydantic import BaseModel
from typing import Dict, Optional, List
from enum import Enum
from runnable_hub import RunnableValueDefine

class AgentFunctionType(str, Enum):
    TOOL = "TOOL"
    AGENT = "AGENT"

class AgentFunction(BaseModel):
    type: AgentFunctionType
    name: str
    version: str
    presetInputs: Dict = {}
