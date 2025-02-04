

from runnable import RunnableResponse
from typing import Dict, Optional

class ChainResponse(RunnableResponse):
    runnableCode: str = "CHAIN_WORKER"
    stdout: str
    stderr: str
    returncode: int | None
    outputs: Dict[str, str] = {}
