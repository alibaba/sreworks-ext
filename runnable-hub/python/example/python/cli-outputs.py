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
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

requestYaml = """
    run: |
      import os
      import sys
      h = open(os.environ.get('PYTHON_RUN_PATH')+"/aaa.txt", "w")
      h.write("hello")
      h.close()
    outputs:
      message: aaa.txt
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