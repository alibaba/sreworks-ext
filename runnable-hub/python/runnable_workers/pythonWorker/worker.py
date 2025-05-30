
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from .request import PythonRequest
from .response import PythonResponse
import sys

import asyncio
import shutil
import os

class Worker(RunnableWorker):

    runnableCode = "PYTHON"
    Request = PythonRequest
    Response = PythonResponse

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

    async def onNext(self, context: RunnableContext[PythonRequest, PythonResponse]) -> RunnableContext:
        
        # create a temporary path to store script
        temporary_path = os.path.join(self.storePath, context.executeId)
        if not os.path.exists(temporary_path):
            os.makedirs(temporary_path)
        
        try:
            temp_file = f"{temporary_path}/run.py"
            with open(temp_file, "w") as h:
                h.write(context.request.run)

            for fileName, fileContent in context.request.data.items():
                if context.request.outputs is not None and fileName in context.request.outputs:
                    raise Exception(f"Data file name '{fileName}' cannot be the same as output file name")
                with open(f"{temporary_path}/{fileName}", "w") as h:
                    h.write(fileContent)

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



