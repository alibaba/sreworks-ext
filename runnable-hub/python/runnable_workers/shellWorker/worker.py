
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from .request.shellRequest import ShellRequest
from .response import ShellResponse

import asyncio
import tempfile
import shutil
import os

class Worker(RunnableWorker):

    runnableCode = "SHELL"
    Request = ShellRequest
    Response = ShellResponse

    shellBin = "/bin/bash"

    def __init__(self, storePath = "/tmp/shell"):
        self.storePath = storePath
        if not os.path.exists(self.storePath):
            os.makedirs(self.storePath)

    @staticmethod
    async def run_command(command, cwd=None, env=None):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    async def onNext(self, context: RunnableContext[ShellRequest, ShellResponse]) -> RunnableContext:
        
        # create a temporary path to store script
        temporary_path = os.path.join(self.storePath, context.executeId)
        if not os.path.exists(temporary_path):
            os.makedirs(temporary_path)

        temp_file = f"{temporary_path}/run.sh"
        with open(temp_file, "w") as h:
            h.write(context.request.run)
        
        try:
            returncode, stdout, stderr = await self.run_command(
                [self.shellBin, temp_file], cwd=temporary_path, env={"SHELL_RUN_PATH": temporary_path})
            outputs = {}
            if context.request.outputs is not None:
                for key, fileName in context.request.outputs.items():
                    with open(f"{temporary_path}/{fileName}", "r") as h:
                        outputs[key] = h.read().strip()

            context.response = ShellResponse(returncode=returncode, stdout=stdout, stderr=stderr, outputs=outputs)
            context.status = RunnableStatus.SUCCESS
                
        finally:
            shutil.rmtree(temporary_path)

        return context



