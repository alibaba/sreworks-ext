
from pydantic import BaseModel
# from runnable import RunnableValueDefine
from typing import List


class AgentPromptChainTemplate(BaseModel):
    systemPrompt: str
    userPrompt: str
    completionPattern: str             # Action:{:s}{:action_input}
    completionParserScript: str        # python script