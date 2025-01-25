
from runnable import RunnableRequest
from .apiHttpMethod import ApiHttpMethod
from .apiResultFormat import ApiResultFormat

class ApiRequest(RunnableRequest):
    runnableCode: str = "API_WORKER"
    url: str
    method: ApiHttpMethod = ApiHttpMethod.GET
    resultFormat: ApiResultFormat = ApiResultFormat.JSON