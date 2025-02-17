import json

class Logger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, key):
        return self.kwargs[key]
    
    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def to_message_text(self, direction=None):
        text = ''
        message_dict = {}
        if 'cur_node' in self.kwargs:
            text += f"Here are the collected information about module [{self.kwargs['cur_node']}]...\n"

            if 'symptom' in self.kwargs:
                message_dict['SYMPTOM'] = self.kwargs['symptom']
            if 'relations' in self.kwargs and not direction=='isolation' and not direction=='full':
                message_dict['RELATIONS'] = self.kwargs['relations']
            if 'neighbors' in self.kwargs and not direction=='isolation':
                if self.kwargs['neighbors']:
                    message_dict['NEIGHBOR MODULES'] = self.kwargs['neighbors']
                else:
                    message_dict['NEIGHBOR MODULES'] = "The related modules have not provided any diagnosis results, it seems that there is no abnormal behavior of neighboring modules.\n"
                
            text += json.dumps(message_dict, indent=4)
        return text

    def print(self):
        text = self.to_message_text()
        print(text)
