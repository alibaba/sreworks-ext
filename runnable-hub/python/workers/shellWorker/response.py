

from runnable import RunnableResponse
from typing import Dict, Optional

class ShellResponse(RunnableResponse):
    runnableCode: str = "SHELL_WORKER"
    stdout: str
    stderr: str
    returncode: int | None
    outputs: Dict[str, str] = {}
