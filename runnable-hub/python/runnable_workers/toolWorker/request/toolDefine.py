
from pydantic import BaseModel
from runnable_hub import RunnableOutputLoads
from .toolType import ToolType
# from .toolParamSpec import ToolParamSpec
from runnable_hub import RunnableValueDefine
from typing import Dict, List, Optional


class ToolDefine(BaseModel):
    toolCode: str
    toolVersion: str
    toolType: ToolType
    setting: Dict                            # 工具配置
    inputSpec: List[RunnableValueDefine] = []      # 输入变量结构
    outputSpec: List[RunnableValueDefine] = []     # 输出变量结构
    outputsLoads: RunnableOutputLoads = RunnableOutputLoads.TEXT
    outputTemplate: Optional[str] = None     # 输出模板
    description: Optional[str] = None        # 工具描述
