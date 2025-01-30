
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.jinjaRequest import JinjaRequest
from .response import JinjaResponse

import asyncio
import tempfile
import os

class Worker(RunnableWorker):

    runnableCode = "SHELL_WORKER"
    Request = JinjaRequest
    Response = JinjaResponse

    def __init__(self):
        pass

    async def onNext(self, context: RunnableContext[JinjaRequest, JinjaResponse]) -> RunnableContext:
        
        pass
        # # create a temporary path to store script
        # temporary_path = os.path.join(self.storePath, context.executeId)
        # if not os.path.exists(temporary_path):
        #     os.makedirs(temporary_path)

        # temp_file = f"{temporary_path}/run.sh"
        # with open(temp_file, "w") as h:
        #     h.write(context.request.run)
        
        # try:
        #     returncode, stdout, stderr = await self.run_command(
        #         [self.shellBin, temp_file], cwd=temporary_path, env={"SHELL_RUN_PATH": temporary_path})
        #     outputs = {}
        #     if context.request.outputs is not None:
        #         for key, fileName in context.request.outputs.items():
        #             with open(f"{temporary_path}/{fileName}", "r") as h:
        #                 outputs[key] = h.read().strip()

        #     context.response = ShellResponse(returncode=returncode, stdout=stdout, stderr=stderr, outputs=outputs)
        #     context.status = RunnableStatus.SUCCESS
                
        # finally:
        #     os.remove(temp_file)

        # return context



