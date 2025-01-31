#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.jinjaWorker.worker import Worker as JinjaWorker
from workers.jinjaWorker.request.jinjaRequest import JinjaRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

requestYaml = """
    template: |
      {"a": "{{ data.name }}"}
    data: 
      name: test
    outputLoads: JSON
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(JinjaWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(JinjaRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())