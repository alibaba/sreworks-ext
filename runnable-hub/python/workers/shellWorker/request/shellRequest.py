
from runnable import RunnableRequest
from typing import Dict, Optional

class ShellRequest(RunnableRequest):
    runnableCode: str = "SHELL_WORKER"
    run: str
    outputs: Optional[Dict[str, str]]          # { "variable name": "output file path" }