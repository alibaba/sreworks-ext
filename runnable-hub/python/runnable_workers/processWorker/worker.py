
from datetime import datetime
from typing import Dict, Any, List
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus
from .request.processRequest import ProcessRequest
from .request.processStep import ProcessStep
from .response import ProcessResponse
from jinja2 import Environment
from uuid import uuid4
import json

class OutputRender:
    def __init__(self, pathValueTemplate):
        self.outputValueMap = {}
        self.pathValueTemplate = pathValueTemplate
    def __getattr__(self, name):
        self.outputValueMap[name] = str(uuid4())[:8]+".txt"
        return {"path": self.pathValueTemplate.format(outputFileName=self.outputValueMap[name])}

def save_filter(value, target_dict):
    target_dict["save"] = value
    return value

class Worker(RunnableWorker):

    runnableCode = "PROCESS"
    Request = ProcessRequest
    Response = ProcessResponse

    def __init__(self):
        self.jinjaNewEnv = Environment(
            variable_start_string="${{",  # 使用 ${{ 作为变量开始标记
            variable_end_string="}}",     # 使用 }} 作为变量结束标记
        )
        self.jinjaNewEnv.filters["save"] = save_filter

    def complexRender(self, data: Dict, target):
        if isinstance(target, dict):
            return {self.complexRender(data, key): self.complexRender(data, value) for key, value in target.items()}
        elif isinstance(target, list):
            return [self.complexRender(data, value) for value in target]
        elif isinstance(target, str):
            target = target.strip()
            # 如果整个字符串只有 ${{ ... }} 则直接取里面的内容，保留变量类型
            if target.startswith("${{") and target.endswith("}}"):
                fetchValue = {}
                self.jinjaNewEnv.from_string(target[:-2] + "|save(fetchValue)}}").render(fetchValue=fetchValue, **data)
                return fetchValue["save"]
            else:
                return self.jinjaNewEnv.from_string(target).render(**data)
        else:
            return target

    def make_worker_request(self, step, stepOutputs) -> Dict:

        # stepRender:Any = self.complexRender({"outputs": outputs, "steps": stepOutputs}, step)
        if step.get("shell") is not None:
            outputs = OutputRender('$SHELL_RUN_PATH/{outputFileName}')
            stepRender:Any = self.complexRender({"outputs": outputs, "steps": stepOutputs}, step)
            stepRender["request"] = {
                "runnableCode": "SHELL",
                "run": stepRender["shell"],
                "outputs": outputs.outputValueMap,
            }
        elif step.get("api") is not None:
            stepRender:Any = self.complexRender({"steps": stepOutputs}, step)
            stepRender["request"] = stepRender["api"]
            stepRender["request"]["runnableCode"] = "API"
        elif step.get("jinja") is not None:
            stepRender:Any = self.complexRender({"steps": stepOutputs}, step)
            stepRender["request"] = stepRender["jinja"]
            stepRender["request"]["runnableCode"] = "JINJA"
        elif step.get("python") is not None:
            outputs = OutputRender("os.environ.get('PYTHON_RUN_PATH')+'/{outputFileName}'")
            stepRender:Any = self.complexRender({"outputs": outputs, "steps": stepOutputs}, step)
            stepRender["request"] = {
                "runnableCode": "PYTHON",
                "run": stepRender["python"],
                "outputs": outputs.outputValueMap,
            }
            stepRender["request"]["runnableCode"] = "PYTHON"
        else:
            stepRender:Any = self.complexRender({"steps": stepOutputs}, step)
            stepRender["request"]["runnableCode"] = stepRender["runnableCode"]
        return stepRender


    async def onNext(self, context: RunnableContext[ProcessRequest, ProcessResponse]) -> RunnableContext:
        
        if context.data.get("runtime") is None:
            context.data["runtime"] = {}
            for jobId, job in context.request.jobs.items():
                if len(job.steps) == 0:
                    raise RuntimeError(f"jobId {jobId} steps is empty")
                context.data["runtime"][jobId] = {
                    "needs": job.needs[:],
                    "steps": [step.model_dump() for step in job.steps],
                    "stepOutputs": {},
                    "errors": {},
                    "currentStepId": None,
                    "startTime": datetime.now(),
                    "outputs": job.outputs,
                }
                if len(job.needs) == 0:
                    context.data["runtime"][jobId]["jobStatus"] = "RUNNING"
                    context.data["runtime"][jobId]["stepStatus"] = "RUNNING"
                else:
                    context.data["runtime"][jobId]["jobStatus"] = "PENDING"
                    context.data["runtime"][jobId]["stepStatus"] = "PENDING"

        for jobId,job in context.data["runtime"].items():
            if job["jobStatus"] != "RUNNING":
                continue

            if context.promise.result.get(jobId) is not None:
                job["stepOutputs"][job["currentStepId"]] = context.promise.result[jobId]
                job["currentStepId"] = None
                job["stepStatus"] = "SUCESS"
            
            elif context.promise.reject.get(jobId) is not None:
                job["errors"][job["currentStepId"]] = context.promise.reject[jobId]
                job["currentStepId"] = None
                job["stepStatus"] = "ERROR"

            if job["currentStepId"] is not None: # job not finish
                continue

            if job["stepStatus"] == "ERROR":
                job["jobStatus"] = "ERROR"
                job["endTime"] = datetime.now()
                for checkJob in context.data["runtime"].values():
                    checkJob["jobStatus"] = "ERROR"
                    checkJob["endTime"] = datetime.now()
            elif len(job["steps"]) > 0:
                step = self.make_worker_request(job["steps"].pop(0), job["stepOutputs"])
                job["currentStepId"] = step["id"]
                context.promise.resolve[jobId] = step["request"]
            else:
                job["jobStatus"] = "SUCCESS"
                job["endTime"] = datetime.now()
                if job["outputs"] is not None:
                    job["outputs"] = self.complexRender({"steps": job["stepOutputs"]}, job["outputs"])
                for checkJob in context.data["runtime"].values():
                    if jobId in checkJob["needs"]:
                        checkJob["needs"].remove(jobId)
                        if len(checkJob["needs"]) == 0 and checkJob["jobStatus"] == "PENDING":
                            checkJob["jobStatus"] = "RUNNING"
                            checkJob["stepStatus"] = "RUNNING"

        allStatus = set([r["jobStatus"] for r in context.data["runtime"].values()])
        if allStatus == set(["SUCCESS"]):
            context.status = RunnableStatus.SUCCESS
            if context.request.outputs is not None:
                jobOutputs = {jobId: {"outputs":job["outputs"]} for jobId, job in context.data["runtime"].items()}
                processOutputs = self.complexRender({"jobs": jobOutputs}, context.request.outputs)
                context.response = ProcessResponse(outputs=processOutputs) # type: ignore

        elif allStatus == set(["ERROR"]) or allStatus == set(["ERROR","SUCCESS"]):
            context.status = RunnableStatus.ERROR

        return context


