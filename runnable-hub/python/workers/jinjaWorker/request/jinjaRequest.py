
from runnable import RunnableRequest, RunnableOutputLoads
from typing import Dict, Optional, List

class JinjaRequest(RunnableRequest):
    runnableCode: str = "JINJA_WORKER"
    data: Dict[str, Dict|List|str|float|int|bool] = {}
    template: str
    outputLoads: RunnableOutputLoads = RunnableOutputLoads.TEXT