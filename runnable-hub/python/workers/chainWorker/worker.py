
from calendar import c
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.chainRequest import ChainRequest
from .request.chainFunction import ChainFunctionType
from .response import ChainResponse
import sys
from jinja2 import Environment

import asyncio
import shutil
import os
import re
import json

class Worker(RunnableWorker):

    runnableCode = "CHAIN_WORKER"
    Request = ChainRequest
    Response = ChainResponse

    pythonBin = sys.executable

    def __init__(self, storePath = "/tmp/python"):
        self_dir = os.path.dirname(__file__)
        self.jinjaEnv = Environment()
        with(open(self_dir + "/next.py", "r")) as h:
            self.nextPostScript = h.read()

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
                "nextStep": "llmEnd",
                "lastCompletion": None,
            }
            return context
        
        elif context.data["runtime"]["nextStep"] == "llmEnd":
            
            if context.promise.result["llm"] is None:
                raise ValueError("LLM response is missing")
            
            context.promise.resolve["processMessage"] = {
                "runnableCode": "PYTHON_WORKER",
                "data": {
                    "completion": context.promise.result["llm"]["content"],
                },
                "run": context.request.onNext + "\n" + self.nextPostScript,
            }
            context.data["runtime"]["nextStep"] = "functionCallStart"
            context.data["runtime"]["lastCompletion"] = context.promise.result["llm"]["content"]
            context.data["runtime"]["history"] += context.promise.result["llm"]["messages"]
            return context

        elif context.data["runtime"]["nextStep"] == "functionCallStart":

            if context.promise.result["processMessage"] is None:
                raise ValueError("processMessage response is missing")

            stdout = context.promise.result["processMessage"]["stdout"]
            matches = re.findall(r"<finalAnswer>(.*?)</finalAnswer>", stdout, re.DOTALL)
            if len(matches) > 0:
                context.status = RunnableStatus.SUCCESS
                context.response = ChainResponse(finalAnswer=matches[0])
                return context
            
            matches = re.findall(r"<function>(.*?)</function>", stdout, re.DOTALL)
            if len(matches) > 0:
                fnCall = json.loads(matches[0])
                fnDefines = [fn for fn in context.request.functions if fn.name == fnCall["name"]]
                if len(fnDefines) == 0:
                    context.status = RunnableStatus.ERROR
                    context.errorMessage = "Function not found"
                    return context
                
                fnDefine = fnDefines[0]
                if fnDefine.type == ChainFunctionType.TOOL:
                    inputs = fnDefine.presetInputs.copy()
                    inputs.update(fnCall["input"])
                    context.promise.resolve["functionCall"] = {
                        "runnableCode": "TOOL_WOKRER",
                        "toolCode": fnDefine.name,
                        "toolVersion": fnDefine.version,
                        "inputs": inputs,
                    }
                elif fnDefine.type == ChainFunctionType.AGENT:
                    inputs = fnDefine.presetInputs.copy()
                    inputs.update(fnCall["input"])
                    context.promise.resolve["functionCall"] = {
                        "runnableCode": "AGENT_WOKRER",
                        "toolCode": fnDefine.name,
                        "toolVersion": fnDefine.version,
                        "inputs": inputs,
                    }
                context.data["runtime"]["nextStep"] = "functionCallEnd"
                return context
            
            context.status = RunnableStatus.ERROR
            context.errorMessage = "finalAnswer or function not found"
            return context

        elif context.data["runtime"]["nextStep"] == "functionCallEnd":

            if context.promise.result["functionCall"] is None:
                raise ValueError("functionCall response is missing")
            
            context.promise.resolve["processMessage"] = {
                "runnableCode": "PYTHON_WORKER",
                "data": {
                    "function": json.dumps(context.promise.result["functionCall"]["outputs"]),
                    "completion": context.data["runtime"]["lastCompletion"],
                },
                "run": context.request.onNext + "\n" + self.nextPostScript,
            }

            context.data["runtime"]["nextStep"] = "llmStart"
            return context

        elif context.data["runtime"]["nextStep"] == "llmStart":

            if context.promise.result["processMessage"] is None:
                raise ValueError("processMessage response is missing")

            stdout = context.promise.result["processMessage"]["stdout"]
            matches = re.findall(r"<message>(.*?)</message>", stdout, re.DOTALL)
            if len(matches) > 0:
                context.promise.resolve["llm"] = {
                    "runnableCode": "LLM_WORKER",
                    "endpoint": context.request.llmEndpoint,
                    "model": context.request.llmModel,
                    "secretKey": context.request.llmSecretKey,
                    "systemPrompt": context.data["runtime"]["systemPrompt"],
                    "userPrompt": matches[0],
                    "history": context.data["runtime"]["history"],
                }
                context.data["runtime"]["nextStep"] = "llmEnd"
                return context
            
            context.status = RunnableStatus.ERROR
            context.errorMessage = "message not found"
            return context 
        
        return context   
            



