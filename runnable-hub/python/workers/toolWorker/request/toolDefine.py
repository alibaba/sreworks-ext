
from pydantic import BaseModel
from .toolType import ToolType
from .toolParamSpec import ToolParamSpec
from typing import Dict, List, Generic, TypeVar, Optional

T = TypeVar('T')

class ToolDefine(BaseModel, Generic[T]):
    toolCode: str
    toolVersion: str
    toolType: ToolType
    setting: T                               # 工具配置
    inputSpec: List[ToolParamSpec] = []      # 输入变量结构
    outputSpec: List[ToolParamSpec] = []     # 输出变量结构
    outputs: Optional[List|Dict]