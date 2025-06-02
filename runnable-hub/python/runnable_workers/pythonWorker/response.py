

from runnable_hub import RunnableResponse
from typing import Dict, Optional

class PythonResponse(RunnableResponse):
    runnableCode: str = "PYTHON"
    stdout: str
    stderr: str
    returncode: int | None
    outputs: Dict[str, str] = {}
