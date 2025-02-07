
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.chainRequest import ChainRequest
from .response import ChainResponse
import sys
from jinja2 import Environment

import asyncio
import shutil
import os

class Worker(RunnableWorker):

    runnableCode = "CHAIN_WORKER"
    Request = ChainRequest
    Response = ChainResponse

    pythonBin = sys.executable

    def __init__(self, storePath = "/tmp/python"):
        self.jinjaEnv = Environment()

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

    async def onNext(self, context: RunnableContext[ChainRequest, ChainResponse]) -> RunnableContext:

        if context.data.get("runtime") is None:
            # todo merge function call defines
            renderData = context.request.data

            systemPrompt = self.jinjaEnv.from_string(context.request.systemPrompt).render(**renderData)
            userPrompt = self.jinjaEnv.from_string(context.request.userPrompt).render(**renderData)

            context.promise.resolve["llm"] = {
                "runnableCode": "LLM_WORKER",
                "endpoint": context.request.llmEndpoint,
                "model": context.request.llmModel,
                "secretKey": context.request.llmSecretKey,
                "systemPrompt": systemPrompt,
                "userPrompt": userPrompt,
            }

            context.data["runtime"] = {
                "history": [],
                "systemPrompt": systemPrompt,
                "nextStep": "interpreter",
                "nextInput": None,
            }
        
        if context.data["runtime"]["currentStep"] == "interpreter":
            if context.promise.result["llm"] is None:
                raise ValueError("LLM response is missing")
            context.promise.resolve["interpreter"] = {
                "runnableCode": "PYTHON_WORKER",
                "data": {
                    "completion": context.promise.result["llm"]["content"],
                },
                "run": context.request.chainInterpreter,
            }            


        return context



