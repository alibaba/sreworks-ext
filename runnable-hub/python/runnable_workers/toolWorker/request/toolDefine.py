
from pydantic import BaseModel
from runnable_hub import RunnableOutputLoads
from .toolType import ToolType
from .toolParamSpec import ToolParamSpec
from typing import Dict, List, Optional


class ToolDefine(BaseModel):
    toolCode: str
    toolVersion: str
    toolType: ToolType
    setting: Dict                            # 工具配置
    inputSpec: List[ToolParamSpec] = []      # 输入变量结构
    outputSpec: List[ToolParamSpec] = []     # 输出变量结构
    outputsLoads: RunnableOutputLoads = RunnableOutputLoads.TEXT
    outputTemplate: Optional[str] = None     # 输出模板
