from ..agent_base import AgentBase

class Critic(AgentBase):
    def __init__(self, name, use_memory, model_config_name,memory_config, system_instructions, user_message, tools=None):
        system_message = system_instructions
        self.user_message = user_message
        super().__init__(
            name,
            system_message,
            model_config_name,          
            use_memory,               
            memory_config,   
            tools
        )
    
    def reply(self, system_info):
        user_message = self.user_message.format(system_introduction=system_info.get("system_introduction"), 
                                                     module_list=system_info.get("module_list"), 
                                                     module_analysis=system_info.get("module_analysis"), 
                                                     analysis_results=system_info.get("analysis_results"))
        # print(user_message)
        message = [{"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": user_message}]

        response = self._reply(message)

        if response:
            return response['choices'][0]["message"]["content"]
        else:
            return "Fail to analyze the abnormal behavior of the system."
