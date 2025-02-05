
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.chainThoughtRequest import ChainThoughtRequest
from .response import ChainThoughtResponse
import sys

import asyncio
import shutil
import os

class Worker(RunnableWorker):

    runnableCode = "CHAIN_THOUGHT_WORKER"
    Request = ChainThoughtRequest
    Response = ChainThoughtResponse

    pythonBin = sys.executable

    def __init__(self, storePath = "/tmp/python"):
        self.storePath = storePath
        if not os.path.exists(self.storePath):
            os.makedirs(self.storePath)

    @staticmethod
    async def run_python(command, cwd=None, env=None):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    async def onNext(self, context: RunnableContext[ChainThoughtRequest, ChainThoughtResponse]) -> RunnableContext:
        
        # create a temporary path to store script
        temporary_path = os.path.join(self.storePath, context.executeId)
        if not os.path.exists(temporary_path):
            os.makedirs(temporary_path)

        temp_file = f"{temporary_path}/run.py"
        with open(temp_file, "w") as h:
            h.write(context.request.run)
        
        try:
            returncode, stdout, stderr = await self.run_python(
                [self.pythonBin, temp_file], cwd=temporary_path, env={"PYTHON_RUN_PATH": temporary_path})
            outputs = {}
            if context.request.outputs is not None:
                for key, fileName in context.request.outputs.items():
                    with open(f"{temporary_path}/{fileName}", "r") as h:
                        outputs[key] = h.read().strip()

            context.response = PythonResponse(returncode=returncode, stdout=stdout, stderr=stderr, outputs=outputs)
            context.status = RunnableStatus.SUCCESS
                
        finally:
            shutil.rmtree(temporary_path)

        return context



