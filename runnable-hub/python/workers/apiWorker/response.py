
from runnable import RunnableResponse

class ApiResponse(RunnableResponse):
    runnableCode: str = "API_WORKER"
    data: object
    