
from runnable import RunnableRequest

class ShellRequest(RunnableRequest):
    runnableCode: str = "SHELL_WORKER"
    run: str