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
    chainInterpreter: |
      import os
      from parse import Parser, with_pattern

      with open(os.environ.get('PYTHON_RUN_PATH')+"/completion", "r") as h:
        completion = h.read()

      def runnable(json_object):
        return "<runnable>" + json.dumps(json_object) + "</runnable>"

      def finalAnswer(text):
        return "<finalAnswer>" + text + "</finalAnswer>"

      @with_pattern(r"\{.*?(?=\n\w+:|\Z)")
      def process_action(text: str) -> dict:
        action = json.loads(text)
        if action.get("action") == "Final answer":
          return finalAnswer(action["action_input"])
        elif action.get("action_input") is not None:
          return f"Observation: {runnable(action)}"

      pattern = "Action:{:s}```{:s}{action:process_action}{:s}```"

      for result in Parser(pattern, {"process_action": process_action}).findall(completion):
        print(result.named.get("action", ""))


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