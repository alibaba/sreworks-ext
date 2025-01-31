#!/usr/bin/env python3

import asyncio
import os
import sys
import yaml
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

from workers.shellWorker.worker import Worker as ShellWorker
from workers.shellWorker.request.shellRequest import ShellRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalFileStore

requestYaml = """
    run: |
        uptime
        sleep 5
        echo "Hello World"
        uptime
"""

async def main():

    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ShellWorker())
    print(runnableHub.workers)

    runnableContext = await runnableHub.executeStart(ShellRequest.model_validate_json(json.dumps(yaml.safe_load(requestYaml))))
    runnableContext = await runnableHub.executeWait(runnableContext)

    print(json.dumps(json.loads(runnableContext.model_dump_json()), indent=4))


if __name__ == "__main__":
    asyncio.run(main())