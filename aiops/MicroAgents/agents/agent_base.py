from __future__ import annotations

import os
from typing import Callable, Dict, Literal, Optional, Union
from functools import partial

from abc import ABCMeta
from typing import Optional
from typing import Sequence
from typing import Union
from typing import Any
from loguru import logger
from utils.load import load_model_by_config_name
from utils.json import parse_json_markdown
# from tools.utils import  convert_func_to_qwen_api
import asyncio
import time

class AgentBase:
    """
    所有agent需要继承这个类, 并实现reply方法 
    """

    _version: int = 1

    def __init__(
        self,
        name: str,                              # agent 名称
        sys_prompt: Optional[str] = None,       # profiling prompt
        model = 'qwen',
        use_memory: bool = False,                # 是否使用memory
        memory_config: Optional[dict] = None,   # memory 配置
        tools = None
    ) -> None:

        self.name = name
        self.memory_config = memory_config

        # agent 的system prompt
        self.sys_prompt = sys_prompt

        self.name_to_tools = {}
        if tools is not None:
            for tool in tools:
                self.name_to_tools[tool.__name__] = tool
        self.tools = tools

        # 根据模型配置文件读取模型， 存放在common configs路径下， 参考backbone.json，定义了temperature等参数
        if model is not None:
            self.model = load_model_by_config_name(model=model)
        else:
            self.model = None


    
    def is_function_call(self, x: str):
        '''
        判断是否是函数调用
        '''
    
        return "tool_calls" == x.choices[0].finish_reason

    def parse_function_call(self, x: str):
        '''
        解析函数调用
        '''
        tool = x.choices[0].message.tool_calls[0].function

        return tool['name'], parse_json_markdown(tool['arguments'])

    
    def invoke_tools(self, tool_name: list, **kwargs):
        '''
        调用工具
        '''
        print("invoke tools: ", tool_name, '\n')
        
        tool = self.name_to_tools[tool_name]
        res = None
        try:
            res = tool(**kwargs)
        except Exception as e:
            res = str(e)
        return res
    
    async def _areply(self, message, max_turns: int = 5, **kwargs):
        tools = []
        if self.tools:
            for tool in self.tools:
                tools.append(convert_func_to_qwen_api(tool))

        for i in range(max_turns):
            response, status = await self.model.areply(message, tools=tools, **kwargs)
            if not status:
                time.sleep(10)
                continue
            
            if self.is_function_call(response):
                message.append(response.choices[0].message)
                tool_name, kwargs = self.parse_function_call(response)
                observation = self.invoke_tools(tool_name, **kwargs)
                message.append({'role':'tool', 'name':tool_name, 'content':observation})
            else:
                break
        if not status:
            print("wrong response from backbone LLM!")
            print("The message is:", message)
            print("The response is:", response)
            return None

        if self.is_function_call(response):
            print( "Too many turns for function call! ")
            return None
    
        return response


    

    def _reply(self, message, max_turns: int = 5, **kwargs) :
        ''' 
             调用LLM，并判断LLM是否进行了工具调用，如果调用工具的话则直接调用工具并进一步问询大模型，直到给出非工具调用的结果。
             同时如果LLM回复报错，statur=False， 则尝试重新问询LLM。
        '''
        
        tools = []
        if self.tools:
            for tool in self.tools:
                tools.append(convert_func_to_qwen_api(tool))

        for i in range(max_turns):
            response, status = self.model.reply(message, tools=tools, **kwargs)
            if not status:
                continue
            
            if self.is_function_call(response):
                message.append(response['choices'][0]['message'])
                tool_name, kwargs = self.parse_function_call(response)
                observation = self.invoke_tools(tool_name, **kwargs)
                message.append({'role':'tool', 'name':tool_name, 'content':observation})
            else:
                break
        if not status:
            print("wrong response from backbone LLM!")
            print("The message is:", message)
            print("The response is:", response)
            return None

        if self.is_function_call(response):
            print( "Too many turns for function call! ")
            return None
    
        return response

    def reply(self, message, max_turns, **kwargs):
        '''
            子类需要实现reply方法，默认直接调用_reply方法
        '''

        return self._reply(message, max_turns, **kwargs)
    


   