

from runnable import RunnableResponse
from typing import Dict, Optional

class PythonResponse(RunnableResponse):
    runnableCode: str = "PYTHON_WORKER"
    stdout: str
    stderr: str
    returncode: int | None
    outputs: Dict[str, str] = {}
