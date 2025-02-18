#!/usr/bin/env python3

from fastapi import FastAPI
import uvicorn
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

# from workers.processWorker.worker import Worker as ProcessWorker
from workers.apiWorker.worker import Worker as ApiWorker
from workers.shellWorker.worker import Worker as ShellWorker
from workers.processWorker.worker import Worker as ProcessWorker
from workers.apiWorker.request.apiRequest import ApiRequest
from workers.shellWorker.request.shellRequest import ShellRequest
from workers.processWorker.request.processRequest import ProcessRequest
from runnable_hub import RunnableHub
from runnable_hub.store import RunnableLocalFileStore


app = FastAPI()


@app.post("/API")
async def apiWorker(request: ApiRequest):
    context = await app.state.runnableHub.executeStart(request)
    return {
        "request": context.request,
        "executeId": context.executeId
    }

@app.post("/SHELL")
async def shellWorker(request: ShellRequest):
    context = await app.state.runnableHub.executeStart(request)
    return {
        "request": context.request,
        "executeId": context.executeId
    }

@app.post("/PROCESS")
async def processWorker(request: ProcessRequest):
    context = await app.state.runnableHub.executeStart(request)
    return {
        "request": context.request,
        "executeId": context.executeId
    }


@app.on_event("startup")
async def startup_event():
    runnableHub = RunnableHub(store=RunnableLocalFileStore("/tmp/"))
    runnableHub.registerWorker(ApiWorker())
    runnableHub.registerWorker(ShellWorker())
    runnableHub.registerWorker(ProcessWorker())
    print(runnableHub.workers)
    app.state.runnableHub = runnableHub


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)