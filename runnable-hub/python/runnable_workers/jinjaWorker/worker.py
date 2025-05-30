
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus, RunnableOutputLoads
from .request.jinjaRequest import JinjaRequest
from .response import JinjaResponse
from jinja2 import Environment
import json

class Worker(RunnableWorker):

    runnableCode = "JINJA"
    Request = JinjaRequest
    Response = JinjaResponse

    def __init__(self):
        self.jinjaEnv = Environment()


    async def onNext(self, context: RunnableContext[JinjaRequest, JinjaResponse]) -> RunnableContext:
        
        renderData = context.request.data

        rawResult = self.jinjaEnv.from_string(context.request.template).render(**renderData)
        if context.request.outputLoads == RunnableOutputLoads.TEXT:
            context.response = JinjaResponse(result=rawResult, outputs=rawResult)
            context.status = RunnableStatus.SUCCESS
        elif context.request.outputLoads == RunnableOutputLoads.JSON:
            context.response = JinjaResponse(result=rawResult, outputs=json.loads(rawResult))
            context.status = RunnableStatus.SUCCESS
        else:
            context.status = RunnableStatus.ERROR
            context.errorMessage = "resultFormat not support"
        
        return context




