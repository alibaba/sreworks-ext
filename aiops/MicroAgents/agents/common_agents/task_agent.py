from ..agent_base import AgentBase   

class TaskAgent(AgentBase):
    def __init__(self, name, use_memory, model,memory_config, system_instructions, user_message, tools=None):
        system_message = system_instructions
        self.user_message = user_message
        super().__init__(
            name,
            system_message,
            model,          
            use_memory,               
            memory_config,   
            tools
        )

    def reply(self, system_info, critic_info=[],externel_message=""):
        user_message = self.user_message.format(system_introduction=system_info.get("system_introduction"), 
                                                     module_list=system_info.get("module_list"), 
                                                     module_analysis=system_info.get("module_analysis"), 
                                                     similar_incidents=system_info.get("similar_incidents"))
        if externel_message:
            user_message += "\n" + externel_message
        message = [{"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": user_message}]

        if critic_info:
            message.extend(critic_info)
            
        # print(message[-1]['content'])
        response = self._reply(message)

        if response:
            return response.choices[0].message.content
        else:
            return "Fail to analyze the abnormal behavior of the system."

    
    def extract_results(self, response):
        keys = ["SYMPTOM", "ABNORMAL", "ABNORMAL_REASON", "CAUSE", "MITIGATION_STEPS"]
        results = {}
        for key in keys:
            results[key] = None
        for line in response.split("\n"):
            for key in keys:
                tmp = line.find(key)
                if tmp >=0:
                    results[key] = line[tmp+len(key)+1:].strip()
        return results


class GroupLeader(AgentBase):
    
    def __init__(self, name, use_memory, model, memory_config, system_instructions, user_message, tools=None):
        system_message = system_instructions
        self.user_message = user_message
        super().__init__(
            name,
            system_message,
            model,          
            use_memory,               
            memory_config,  
            tools
        )

    def reply(self, analysis_results):
        user_message = self.user_message.format(analysis_results=analysis_results)
        message = [{"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": user_message}]
        response = self._reply(message)
        if response:
            return response.choices[0].message.content
        else:
            return "Fail to analyze the abnormal behavior of the system."

    def extract_results(self, response):
        keys = ["SYMPTOM", "ABNORMAL", "ABNORMAL_REASON", "CAUSE", "MITIGATION_STEPS"]
        results = {}
        for key in keys:
            results[key] = None
        for line in response.split("\n"):
            for key in keys:
                tmp = line.find(key)
                if tmp >=0:
                    results[key] = line[tmp+len(key)+1:].strip()
        return results