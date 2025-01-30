
from pydantic import BaseModel
from .toolType import ToolType
from .toolTemplate import ToolTemplate
from .toolParamSpec import ToolParamSpec
from typing import Dict, List

class ToolDefine(BaseModel):
    toolCode: str
    toolVersion: str
    toolType: ToolType
    setting: Dict                       # 工具配置
    inputSpec: List[ToolParamSpec]      # 输入变量结构
    outputSpec: List[ToolParamSpec]     # 输出变量结构
    template: ToolTemplate              # 模板集合