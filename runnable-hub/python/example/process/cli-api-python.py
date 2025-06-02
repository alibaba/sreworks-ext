#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from runnable_workers.processWorker.worker import Worker as ProcessWorker
from runnable_workers.processWorker.request.processRequest import ProcessRequest
from runnable_workers.apiWorker.worker import Worker as ApiWorker
# from runnable_workers.jinjaWorker.worker import Worker as JinjaWorker
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

QWEN_SK = os.getenv("QWEN_SK")

requestYaml = f"""
    jobs:
      api:
        steps:
        - id: api
          api: 
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
                  content: |
                    请帮我使用python的正则匹配json，返回的python代码请放在
                    Action: 
                    {{
                      "action":"python",
                      "action_input":"..."
                    }}
                    这样的格式中

    
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(ApiWorker())
    # runnableHub.registerWorker(JinjaWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(
        ProcessRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())