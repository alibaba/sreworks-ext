#!/usr/bin/env python3

from fastapi import FastAPI
import uvicorn

# from workers.processWorker.worker import Worker as ProcessWorker
from workers.apiWorker.worker import Worker as ApiWorker
from workers.apiWorker.request.apiRequest import ApiRequest
from workers.processWorker.request.processRequest import ProcessRequest
from runnable import RunnableHub
from runnable.store import RunnableLocalStore


app = FastAPI()


@app.post("/API")
async def apiWorker(request: ApiRequest):
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
    runnableHub = RunnableHub(store=RunnableLocalStore("/tmp/"))
    runnableHub.registerWorker(ApiWorker())
    app.state.runnableHub = runnableHub


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)