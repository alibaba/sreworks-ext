
from runnable_hub import RunnableRequest, RunnableOutputLoads
from typing import Dict, Optional, List

class JinjaRequest(RunnableRequest):
    runnableCode: str = "JINJA"
    data: Dict[str, Dict|List|str|float|int|bool] = {}
    template: str
    outputLoads: RunnableOutputLoads = RunnableOutputLoads.TEXT