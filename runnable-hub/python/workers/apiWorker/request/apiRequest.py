
from runnable_hub import RunnableRequest, RunnableOutputLoads
from .apiHttpMethod import ApiHttpMethod
from typing import Dict

class ApiRequest(RunnableRequest):
    runnableCode: str = "API"
    url: str
    method: ApiHttpMethod = ApiHttpMethod.GET
    outputLoads: RunnableOutputLoads = RunnableOutputLoads.TEXT
    headers: Dict = {}
    params: Dict = {}
    payloads: Dict = {}