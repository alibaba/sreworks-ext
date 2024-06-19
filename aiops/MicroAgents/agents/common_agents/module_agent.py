
import os
from typing import Callable, Dict, Literal, Optional, Union
# from backbone.chat import alichat
from functools import partial
from ..agent_base import AgentBase         
import networkx as nx
from .expert_agent import ExpertAgent
import copy
import asyncio

class ModuleAgent(AgentBase):
    '''
        用于分析某个module的异常机器根因，可以接受其他module的diagnosis结果辅助分析
    '''
  
    def __init__(self, name: str, module: str, 
        system_instructions: Optional[str] = '',
        model: str = "qwen",
        use_memory: bool = False,
        memory_config: Optional[Dict[str, Union[str, int, float]]] = None, tools=None,
        **kwargs):
        
        self.module = module
        system_message = system_instructions.format(module_name=module["module_name"], module_function=module["module_function"], module_dependency=module["module_dependency"])

        super().__init__(
            name,
            system_message,
            model,          
            use_memory,               
            memory_config,
            tools
        )
    
    async def areply(self, user_message):
        message = [
            {"role":"system", "content": self.sys_prompt},
            {"role": "user", "content": user_message}
        ]
        response= await self._areply(message)
        if response:
            return response.choices[0].message.content
     
        return "Fail to analyze the abnoraml behavior."


    def reply(self, user_message):
        '''
        self_message: 自身的异常信息，如某个指标发生了异常；
        message_from_other_agent: 其他module的diagnosis结果或者异常信息
        '''
        message = [
                {"role":"system", "content": self.sys_prompt},
                {"role": "user", "content": user_message}
            ]
        response=self._reply(message)

        if response:
            return response.choices[0].message.content
     
        return "Fail to analyze the abnoraml behavior."