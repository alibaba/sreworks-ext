
from runnable import RunnableRequest
from typing import Dict, Optional
from .jinjaResultFormat import JinjaResultFormat

class JinjaRequest(RunnableRequest):
    runnableCode: str = "JINJA_WORKER"
    template: str
    resultFormat: JinjaResultFormat = JinjaResultFormat.JSON