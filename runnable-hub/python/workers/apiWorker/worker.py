from httpx import request
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from .request.apiRequest import ApiRequest, ApiResultFormat, ApiHttpMethod
from .response import ApiResponse
import aiohttp

class Worker(RunnableWorker):

    runnableCode = "API_WORKER"
    Request = ApiRequest
    Response = ApiResponse

    def __init__(self):
        pass

    @staticmethod
    async def makeResponse(response, resultFormat: ApiResultFormat):
        if resultFormat == ApiResultFormat.JSON:
            return await response.json()
        else:
            return await response.text()

    async def onNext(self, context: RunnableContext[ApiRequest, ApiResponse]) -> RunnableContext:
        
        if context.request.method == ApiHttpMethod.GET:
            async with aiohttp.ClientSession() as session:
                async with session.get(context.request.url, headers=context.request.headers) as response:
                    context.response = ApiResponse(data=await self.makeResponse(response, context.request.resultFormat))
                    context.status = RunnableStatus.SUCCESS
                    return context
        else:
            context.status = RunnableStatus.ERROR
            context.errorMessage = "method not support"
            return context
    



