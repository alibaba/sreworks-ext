

from runnable_hub import RunnableResponse
from typing import Dict, Optional

class ShellResponse(RunnableResponse):
    runnableCode: str = "SHELL"
    stdout: str
    stderr: str
    returncode: int | None
    outputs: Dict[str, str] = {}
