
from datetime import datetime
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.processRequest import ProcessRequest
from .request.processStep import ProcessStep

class Worker(RunnableWorker):

    runnableCode = "PROCESS_WORKER"
    Request = ProcessRequest

    def __init__(self):
        pass

    def onNext(self, context: RunnableContext[ProcessRequest]) -> RunnableContext:
        
        if context.data.get("runtime") is None:
            context.data["runtime"] = {}
            for job in context.request.jobs:
                if job.jobId in context.data["runtime"]:
                    raise RuntimeError(f"jobId {job.jobId} duplicated")
                if len(job.steps) == 0:
                    raise RuntimeError(f"jobId {job.jobId} steps is empty")
                context.data["runtime"][job.jobId] = {
                    "needs": job.needs,
                    "steps": job.steps,
                    "outputs": {},
                    "currentStepId": None,
                    "startTime": datetime.now(),
                }
                if len(job.needs) == 0:
                    context.data["runtime"][job.jobId]["jobStatus"] = "RUNNING"
                    context.data["runtime"][job.jobId]["stepStatus"] = "RUNNING"
                else:
                    context.data["runtime"][job.jobId]["jobStatus"] = "PENDING"
                    context.data["runtime"][job.jobId]["stepStatus"] = "PENDING"

        finishCount = 0
        existError = False
        for jobId,job in context.data["runtime"].items():
            if job["jobStatus"] == "RUNNING":

                if context.promise.result.get(jobId) is not None:
                    stepResult = context.promise.result[jobId]
                    job["outputs"][job["currentStepId"]] = context.promise.result[jobId]
                    job["currentStepId"] = None
                    job["stepStatus"] = stepResult["status"]

                if job["currentStepId"] is not None: # job not finish
                    continue

                if job["stepStatus"] == "ERROR":
                    job["jobStatus"] = "ERROR"
                    job["endTime"] = datetime.now()
                    for checkJob in context.data["runtime"].values():
                        checkJob["jobStatus"] = "ERROR"
                        checkJob["endTime"] = datetime.now()
                elif len(job["steps"]) > 0:
                    step:ProcessStep = job["steps"].pop(0)
                    step.request["runnableCode"] = step.runnableCode
                    job["currentStepId"] = step.stepId
                    context.promise.resolve[jobId] = step.request
                else:
                    job["jobStatus"] = "SUCCESS"
                    job["endTime"] = datetime.now()
                    for checkJob in context.data["runtime"].values():
                        if jobId in checkJob["needs"]:
                            checkJob["needs"].remove(jobId)
                        if len(checkJob["needs"]) == 0:
                            checkJob["jobStatus"] = "RUNNING"
                            checkJob["stepStatus"] = "RUNNING"

            elif job["jobStatus"] == "SUCCESS":
                finishCount += 1
            elif job["jobStatus"] == "ERROR":
                finishCount += 1
                existError = True
            if job["jobStatus"] == "PENDING":
                pass

        if finishCount == len(context.data["runtime"]):
            if existError:
                context.status = RunnableStatus.ERROR
            else:
                context.status = RunnableStatus.SUCCESS

        return context


