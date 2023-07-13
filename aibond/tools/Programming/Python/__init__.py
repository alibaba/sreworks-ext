#!/usr/bin/python
# -*- coding: UTF-8 -*-

from typing import Dict
import sys
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def execute_python(code_str: str, return_context: bool=False) -> Dict:
    """This is a Python execution tool. You can input a piece of Python code, and it will return the corresponding execution results. By default, it returns the first 1000 characters of both stdout and stderr. It's recommended to use the print() function to directly display the results."""
    # 为输出和错误创建StringIO对象，以便我们可以捕获它们
    stdout = StringIO()
    stderr = StringIO()
    return_head = 1000

    context = {}

    try:
        # 重定向stdout和stderr，执行代码
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code_str, context)
    except Exception:
        stderr.write(traceback.format_exc())

    # 获取执行后的stdout, stderr和context
    stdout_value = stdout.getvalue()[0:return_head]
    stderr_value = stderr.getvalue()[0:return_head]
   
    if return_context == True:
        return {"stdout": stdout_value, "stderr": stderr_value, "context": context}
    else:
       return {"stdout": stdout_value, "stderr": stderr_value, "context": {}}


