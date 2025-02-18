#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.apiWorker.worker import Worker as ApiWorker
from workers.apiWorker.request.apiRequest import ApiRequest
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")

requestYaml = f"""
    url: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
    method: POST
    outputLoads: JSON
    headers:
        Authorization: Bearer {QWEN_SK}
        Content-Type: application/json
    payloads:
        model: qwen-plus
        messages:
        - role: system
          content: You are a helpful assistant.
        - role: user
          content: 请帮我使用python的正则匹配json
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ApiWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(ApiRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())