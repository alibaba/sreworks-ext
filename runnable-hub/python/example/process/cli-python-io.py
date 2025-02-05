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
from workers.pythonWorker.worker import Worker as PythonWorker
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

requestYaml = """
    jobs:
      test:
        steps:
        - id: testStep1
          python: |
            import os
            h = open(${{ outputs.abc.path }}, "w")
            h.write("hello")
            h.close()
        - id: testStep2
          python: |
            print("hhh")
            print('${{ steps.testStep1.outputs.abc }}')
    
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ProcessWorker())
    runnableHub.registerWorker(PythonWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(
        ProcessRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())