
from runnable import RunnableRequest
from typing import Dict, Optional, List
from .jinjaResultFormat import JinjaResultFormat

class JinjaRequest(RunnableRequest):
    runnableCode: str = "JINJA_WORKER"
    data: Dict[str, Dict|List|str|float|int|bool] = {}
    template: str
    resultFormat: JinjaResultFormat = JinjaResultFormat.TEXT