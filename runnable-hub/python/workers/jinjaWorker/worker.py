
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.jinjaRequest import JinjaRequest, JinjaResultFormat
from .response import JinjaResponse
from jinja2 import Environment
import json

class Worker(RunnableWorker):

    runnableCode = "JINJA_WORKER"
    Request = JinjaRequest
    Response = JinjaResponse

    def __init__(self):
        self.jinjaEnv = Environment()


    async def onNext(self, context: RunnableContext[JinjaRequest, JinjaResponse]) -> RunnableContext:
        
        rawResult = self.jinjaEnv.from_string(context.request.template).render(data=context.request.data)
        if context.request.resultFormat == JinjaResultFormat.TEXT:
            context.response = JinjaResponse(result=rawResult)
            context.status = RunnableStatus.SUCCESS
        elif context.request.resultFormat == JinjaResultFormat.JSON:
            context.response = JinjaResponse(result=json.loads(rawResult))
            context.status = RunnableStatus.SUCCESS
        else:
            context.status = RunnableStatus.ERROR
            context.errorMessage = "resultFormat not support"
        
        return context




