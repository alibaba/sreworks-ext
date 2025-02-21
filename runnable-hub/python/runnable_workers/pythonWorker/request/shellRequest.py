
from runnable_hub import RunnableRequest
from typing import Dict, Optional

class ShellRequest(RunnableRequest):
    runnableCode: str = "SHELL"
    run: str
    outputs: Optional[Dict[str, str]] = None          # { "variable name": "output file path" }