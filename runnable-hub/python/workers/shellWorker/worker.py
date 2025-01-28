
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.shellRequest import ShellRequest
from .response import ShellResponse

import asyncio
import tempfile
import os

class Worker(RunnableWorker):

    runnableCode = "SHELL_WORKER"
    Request = ShellRequest
    Response = ShellResponse

    shellBin = "/bin/bash"

    def __init__(self):
        pass

    @staticmethod
    async def run_command(command):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    async def onNext(self, context: RunnableContext[ShellRequest, ShellResponse]) -> RunnableContext:
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(context.request.run.encode())
            temp_file_path = temp_file.name

        try:
            returncode, stdout, stderr = await self.run_command([self.shellBin, temp_file_path])
            context.response = ShellResponse(returncode=returncode, stdout=stdout,stderr=stderr)
            context.status = RunnableStatus.SUCCESS
        finally:
            os.remove(temp_file_path)

        return context



