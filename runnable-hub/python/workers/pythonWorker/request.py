
from runnable import RunnableRequest
from typing import Dict, Optional

class PythonRequest(RunnableRequest):
    runnableCode: str = "PYTHON_WORKER"
    run: str
    outputs: Optional[Dict[str, str]] = None          # { "variable name": "output file path" }