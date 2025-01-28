
from pydantic import BaseModel
from .toolType import ToolType
from typing import Dict

class ToolDefine(BaseModel):
    toolCode: str
    toolVersion: str
    toolType: ToolType
    payload: Dict
    inputSpec: Dict             # 输入变量结构
    outputSpec: Dict            # 输出变量结构
    inputMappingTemplate: str   # 输入变量映射模板,支持jinja2模板
    outputMappingTemplate: str  # 输出变量映射模板,支持jinja2模板