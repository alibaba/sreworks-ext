
def convert_func_to_qwen_api(func):
    
    res = {}
    res["name"] = func.__name__
    res["description"] = func.__doc__
    res["parameters"] = {"type": "object", "properties": {}, "required": []}
    for param in func.params:
        res["parameters"]["properties"][param] = {"type": func.params[param]["type"], "description": func.params[param]["description"]}
        if 'required' in func.params[param] and func.params[param]['required']:
            res["parameters"]["required"] = res["parameters"]["required"] + [param]
    res = {"type":"function", "function":res}
    return res