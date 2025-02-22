import re
import json
from pydantic import BaseModel
from typing import Optional, Dict

class FunctionCall(BaseModel):
    name: str
    input: Dict|str

class ChainDo(BaseModel):
    message: Optional[str] = None             # Message to be replied
    function: Optional[FunctionCall] = None   # Function call to be executed
    finalAnswer: Optional[str] = None         # Final answer to be returned
    overwriteMessage: Optional[str] = None    # Message to be overwritten

class ChainContext(BaseModel):
    message: str
    function: Optional[str]
    do: ChainDo = ChainDo()

def onNext(context: ChainContext):

    if context.function is None:
        pattern = re.compile(r'Action:\s*```(.*?)```', re.DOTALL | re.IGNORECASE)
        match = pattern.search(context.message)
        if match:
            json_str = match.group(1).strip()  # 提取内容并去除首尾空白
            try:
                action_data = json.loads(json_str)
                if "final answer" in action_data.get("action","").lower():
                    context.do.finalAnswer = action_data["action_input"]
                elif action_data.get("action_input") is not None:
                    context.do.function = FunctionCall(
                        name=action_data["action"], 
                        input=action_data["action_input"])
            except json.JSONDecodeError:
                raise Exception("json parse fail, content:" + json_str)
        else:
            raise Exception("No action found")
    else:
        context.do.message = "Observation: " + context.function

    return context