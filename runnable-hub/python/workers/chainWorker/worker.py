
from typing import Dict
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from .request.chainRequest import ChainRequest, ChainFunction
from .request.chainFunction import ChainFunctionType
from .response import ChainResponse
import sys
from jinja2 import Environment

import os
import re
import json


# enter   -->     llmEnd        <----   llmStart
#                    |                      ^ 
#                    |                      |
#                    v                      |    
#               functionCallStart  --->  functionCallEnd  --->  finish
            

class Worker(RunnableWorker):

    runnableCode = "CHAIN"
    Request = ChainRequest
    Response = ChainResponse

    pythonBin = sys.executable

    def __init__(self, storePath = "/tmp/python"):
        self_dir = os.path.dirname(__file__)
        self.jinjaEnv = Environment()
        with(open(self_dir + "/next.py", "r")) as h:
            self.nextPostScript = h.read()

    @staticmethod
    def fnInputMerge(fn: ChainFunction, inputs: Dict|str) -> Dict:
        finalInputs = fn.presetInputs.copy()
        if isinstance(inputs, str):
            finalInputs.update({
                fn.inputDefine[0].name: inputs
            })
        else:
            finalInputs.update(inputs)
        return finalInputs


    # @staticmethod
    # async def run_python(command, cwd=None, env=None):
    #     process = await asyncio.create_subprocess_exec(
    #         *command,
    #         stdout=asyncio.subprocess.PIPE,
    #         stderr=asyncio.subprocess.PIPE,
    #         cwd=cwd,
    #         env=env
    #     )
    #     stdout, stderr = await process.communicate()
    #     return process.returncode, stdout.decode(), stderr.decode()


    async def onNext(self, context: RunnableContext[ChainRequest, ChainResponse]) -> RunnableContext:

        if context.data.get("runtime") is None:
            # todo merge function call defines
            renderData = context.request.data

            renderData["tool_info"] = ""
            renderData["function_info"] = ""
            renderData["agent_info"] = ""
            for fn in context.request.functions:
                args = ", ".join([f"{arg.name}: {arg.type.value}" for arg in fn.inputDefine])
                info_block = f"{fn.name}: {fn.name}({args}) - {fn.description}\n"
                renderData["function_info"] += info_block
                if fn.type == ChainFunctionType.TOOL:
                    renderData["tool_info"] += info_block
                elif fn.type == ChainFunctionType.AGENT:
                    renderData["agent_info"] += info_block

            systemPrompt = self.jinjaEnv.from_string(context.request.systemPrompt).render(**renderData)
            userPrompt = self.jinjaEnv.from_string(context.request.userPrompt).render(**renderData)

            context.promise.resolve["llm"] = {
                "runnableCode": "LLM",
                "setting": context.request.llm.model_dump(),
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
                "runnableCode": "PYTHON",
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
                context.response = ChainResponse(finalAnswer=matches[0],history=context.data["runtime"]["history"])
                return context
            
            matches = re.findall(r"<function>(.*?)</function>", stdout, re.DOTALL)
            if len(matches) > 0:
                fnCall = json.loads(matches[0])
                fnDefines = [fn for fn in context.request.functions if fn.name == fnCall["name"]]
                if len(fnDefines) == 0:
                    context.status = RunnableStatus.ERROR
                    context.errorMessage = f"Function:{fnCall['name']} not found"
                    return context
                
                fnDefine = fnDefines[0]

                if fnDefine.type == ChainFunctionType.TOOL:
                    runnableCode = "TOOL"
                elif fnDefine.type == ChainFunctionType.AGENT:
                    runnableCode = "AGENT"

                context.promise.resolve["functionCall"] = {
                    "runnableCode": runnableCode,
                    "toolCode": fnDefine.name,
                    "toolVersion": fnDefine.version,
                    "inputs": self.fnInputMerge(fnDefine, fnCall["input"]),
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
                "runnableCode": "PYTHON",
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
                    "runnableCode": "LLM",
                    "setting": context.request.llm.model_dump(),
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
            



