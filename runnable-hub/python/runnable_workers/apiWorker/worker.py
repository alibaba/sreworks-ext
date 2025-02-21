import json
from runnable_hub import RunnableWorker, RunnableContext, RunnableStatus, RunnableOutputLoads
from .request.apiRequest import ApiRequest, ApiHttpMethod
from .response import ApiResponse
import aiohttp

class Worker(RunnableWorker):

    runnableCode = "API"
    Request = ApiRequest
    Response = ApiResponse

    def __init__(self):
        pass

    @staticmethod
    def outputs(raw, loadType: RunnableOutputLoads):
        if loadType == RunnableOutputLoads.JSON:
            return json.loads(raw)
        else:
            return raw

    async def onNext(self, context: RunnableContext[ApiRequest, ApiResponse]) -> RunnableContext:
        
        if context.request.method == ApiHttpMethod.GET:
            async with aiohttp.ClientSession() as session:
                async with session.get(context.request.url, 
                                       headers=context.request.headers, params=context.request.params) as response:
                    result = await response.text()
                    context.response = ApiResponse(
                        result=result, outputs=self.outputs(result, context.request.outputLoads),
                        statusCode=response.status)
                    context.status = RunnableStatus.SUCCESS
                    return context
        elif context.request.method == ApiHttpMethod.POST:
            async with aiohttp.ClientSession() as session:
                async with session.post(context.request.url, 
                                        headers=context.request.headers, params=context.request.params, json=context.request.payloads) as response:
                    result = await response.text()
                    context.response = ApiResponse(
                        result=result, outputs=self.outputs(result, context.request.outputLoads),
                        statusCode=response.status)
                    context.status = RunnableStatus.SUCCESS
                    return context
        else:
            context.status = RunnableStatus.ERROR
            context.errorMessage = "method not support"
            return context
    



