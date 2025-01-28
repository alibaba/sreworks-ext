
from runnable import RunnableRequest
from typing import Dict

class ToolRequest(RunnableRequest):
    runnableCode: str = "TOOL_WORKER"
    toolCode: str
    payload: Dict