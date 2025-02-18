#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.processWorker.worker import Worker as ProcessWorker
from workers.processWorker.request.processRequest import ProcessRequest
from workers.apiWorker.worker import Worker as ApiWorker
from workers.shellWorker.worker import Worker as ShellWorker
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

requestYaml = """
    jobs:
      test:
        steps:
        - id: testStep1
          runnableCode: SHELL
          request:
            run: |
              uptime
              sleep 1
              uptime
        - id: testStep2
          runnableCode: SHELL
          request:
            run: |
              uptime
    
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(ShellWorker())
    runnableHub.registerWorker(ApiWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(
        ProcessRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())