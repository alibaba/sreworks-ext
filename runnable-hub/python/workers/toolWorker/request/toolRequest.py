
from runnable_hub import RunnableRequest
from typing import Dict

class ToolRequest(RunnableRequest):
    runnableCode: str = "TOOL"
    toolCode: str
    toolVersion: str
    inputs: Dict