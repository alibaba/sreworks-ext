#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.llmWorker.worker import Worker as LlmWorker
from workers.llmWorker.request.llmRequest import LlmRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")

requestYaml = f"""
    setting:
      endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
      model: qwen-plus
      secretKey: {QWEN_SK}
    systemPrompt: You are a helpful assistant.
    userPrompt: 请帮我使用python的正则匹配json
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(LlmWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(LlmRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())