from httpx import request
from runnable import RunnableWorker, RunnableContext, RunnableStatus
from request.apiRequest import ApiRequest, ApiResultFormat
from response import ApiResponse
import aiohttp

class Worker(RunnableWorker):

    runnableCode = "API_WORKER"

    def __init__(self):
        print("processWorker init")

    @staticmethod
    def make_response(response, resultFormat: ApiResultFormat):
        if resultFormat == ApiResultFormat.JSON:
            return response.json()
        else:
            return response.text()

    async def onNext(self, context: RunnableContext[ApiRequest]) -> RunnableContext:
        # request:ApiRequest = context.request
        if context.request.method == "GET":
            async with aiohttp.ClientSession() as session:
                async with session.get(context.request.url) as response:
                    context.response = ApiResponse(data=await self.make_response(response, context.request.resultFormat))
                    context.status = RunnableStatus.SUCCESS
                    return context
        else:
            context.status = RunnableStatus.ERROR
            context.errorMessage = "method not support"
            return context



