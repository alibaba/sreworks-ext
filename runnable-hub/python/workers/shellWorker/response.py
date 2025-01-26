
from sys import stderr
from runnable import RunnableResponse

class ShellResponse(RunnableResponse):
    runnableCode: str = "SHELL_WORKER"
    stdout: str
    stderr: str
    returncode: int | None
