from functools import wraps



#定义修饰符，用于装饰工具的参数，具体使用参考其他工具
def tool(**info):
    def decorator(func):
        for key in info.keys():
            info[key]['required'] = True if 'required' not in info[key] else info[key]['required']
        func.params = info
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


