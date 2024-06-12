from typing import Callable, Dict, List, Literal, Optional, Union

from ..agent_base import AgentBase

class ExpertAgent(AgentBase):
    '''
    一个简单的agent包装，用于human in loop 模式，提供了人工进行分析的借口
    '''
    def __init__(self, human_input_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.human_input_mode = human_input_mode
        
    def reply(self, observation):
        lines = []
        if self.human_input_mode:
            print(f"Here are the obeservation made by the LLM:\n{observation}")
            print("Please input your analysis based on the observation (input 'exit' or press enter to exit): ")
            while True:
                line = input()
                if line.lower() == "exit" or line == "\n"  or not line:
                    break
                line = line.strip('\n')
                if line:
                    lines.append(line.strip('\n'))
            lines = '\n'.join(lines)
            return lines
        else:
            return None