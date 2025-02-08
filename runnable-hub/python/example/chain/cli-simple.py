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
   
      class FunctionCall(BaseModel):
        name: str
        input: Dict
      
      class ChainDo(BaseModel):
        message: Optional[str]                    # Message to be replied
        function: Optional[FunctionCall]          # Function call to be executed
        finalAnswer: Optional[str]                # Final answer to be returned
        overwriteMessage: Optional[str]           # Message to be overwritten

      class ChainContext(BaseModel):
        message: str
        function: Optional[str]
        do: ChainDo = ChainDo()

      def onNext(context: ChainContext):
      
        if context.function is None:
          pattern = re.compile(r'Action:\s*```\n?(.*?)```', re.DOTALL | re.IGNORECASE)
          match = pattern.search(context.message)
          if match:
            json_str = match.group(1).strip()  # 提取内容并去除首尾空白
            try:
                action_data = json.loads(json_str)
                if action_data.get("action") == "Final answer":
                  context.do.finalAnswer = action_data["action_input"]
                elif action_data.get("action_input") is not None:
                  context.do.function = FunctionCall(
                    name=action_data["action"], 
                    input=action_data["action_input"])
            except json.JSONDecodeError:
              raise Exception("json parse fail, content:\n", json_str)
          else:
            raise Exception("No action found")
        else:
          context.do.message = "Observation: " + context.function

        return context
      

    onNextPost: |
      import os
      with open(os.environ.get('PYTHON_RUN_PATH')+"/completion", "r") as h:
        completion = h.read()
      
      try:
        context = onNext(ChainContext(message=completion))
        if context.do.messageOverwrite is not None:
          print("<overwriteMessage>" + context.do.overwriteMessage + "</overwriteMessage>")
        if context.do.message is not None:
          print("<message>" + context.message + "</message>")
        elif context.do.function is not None:
          print("<function>" + json.dumps(context.do.function.dict()) + "</function>")
        elif context.do.finalAnswer is not None:
          print("<finalAnswer>" + context.do.finalAnswer + "</finalAnswer>")
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