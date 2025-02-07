#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.chainWorker.worker import Worker as ChainWorker
from workers.chainWorker.request.chainRequest import ChainRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")

requestYaml = f"""
    llmEndpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    llmModel: qwen-plus
    llmSecretKey: {QWEN_SK}
"""

requestYaml += """
    functions: []
    systemPrompt: str
    userPrompt: str
    onNext: |
      from pydantic import BaseModel
      from typing import Optional, Dict
      from enum import Enum
      
      class Stage(Enum):
        AFTER_PROMPT = "after_prompt"

      
      class FunctionCall(BaseModel):
        name: str
        inputs: Dict
      
      class ChainContext(BaseModel):
        message: str
        callResult: Optional[str]
        messageReply: Optional[str]               # Message to be replied
        functionCall: Optional[FunctionCall]      # Function call to be executed
        finalAnswer: Optional[str]                # Final answer to be returned
        messageOverwrite: Optional[str]           # Message to be overwritten

      def onNext(context: ChainContext):
      
        if context.callResult is None:
          pattern = re.compile(r'Action:\s*```\n?(.*?)```', re.DOTALL | re.IGNORECASE)
          match = pattern.search(context.message)
          if match:
            json_str = match.group(1).strip()  # 提取内容并去除首尾空白
            try:
                action_data = json.loads(json_str)
                if action_data.get("action") == "Final answer":
                  context.finalAnswer = action_data["action_input"]
                elif action_data.get("action_input") is not None:
                  context.functionCall = FunctionCall(
                    name=action_data["action"], 
                    inputs=action_data["action_input"])
            except json.JSONDecodeError:
              raise Exception("json parse fail, content:\n", json_str)
          else:
            raise Exception("No action found")
        else:
          context.messageReply = "Observation: " + context.callResult

        return context
      

    onNextPost: |
      import os
      with open(os.environ.get('PYTHON_RUN_PATH')+"/completion", "r") as h:
        completion = h.read()
      
      try:
        context = onNext(ChainContext(message=completion))
        if context.messageOverwrite is not None:
          print("<messageOverwrite>" + context.messageOverwrite + "</messageOverwrite>")
        if context.messageReply is not None:
          print("<messageReply>" + context.messageReply + "</messageReply>")
        elif context.functionCall is not None:
          print("<functionCall>" + json.dumps(context.functionCall.dict()) + "</functionCall>")
        elif context.finalAnswer is not None:
          print("<finalAnswer>" + context.finalAnswer + "</finalAnswer>")
      except Exception as e:
        print("<error>"+str(e)+"</error>")
      
      
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ChainWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(ChainRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())