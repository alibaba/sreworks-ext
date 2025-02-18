#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.pythonWorker.worker import Worker as PythonWorker
from workers.pythonWorker.request import PythonRequest
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore

requestYaml = """
    data: 
      test: "hello"
    run: |
      import os
      h = open(os.environ.get('PYTHON_RUN_PATH')+"/test", "r")
      print(h.read())
      h.close()
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(PythonWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(PythonRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())