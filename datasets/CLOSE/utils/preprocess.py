##
# 
#Copyright (c) 2023, Alibaba Group;
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#

import re
import time
import pandas as pd
from collections import OrderedDict

def add_blank_token(log):
    """
        替换日志中的空格符为特殊token标志[blank]
    """
    new_log = re.sub('\s+','[blank]',log)
    return new_log

def add_var_token(rex, line):
    """
        对日志进行正则匹配，将匹配到的变量合并为特殊token标注“[var]”
    """
    for currentRex in rex:
        line = re.sub(currentRex, '[var]', line)
    line = re.sub('(\[var\])+','[var]',line)
    # line = re.sub('(\s*\[var\]\s*)+',' [var] ',line)
    return line

def clean_log(log):
    # 去除特殊字符并保留特殊token“[var]”
    log = replace_special_characters(log)
    #分割驼峰词
    pattern = r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
    log = re.sub(pattern, ' ', log)
    # 合并多个空格字符
    log = re.sub("\s{2,}"," ",log)
    return log

def replace_special_characters(s):
    words = s.split()
    pattern = r"[^A-Za-z0-9]"
    result = []
    for word in words:
        # If [var] is found in the word, replace special characters only in the non-[var] part of the word
        if "[var]" in word:
            parts = word.split("[var]")
            new_parts = [re.sub(pattern, " ", part) for part in parts]
            # print(new_parts)
            new_word = " [var] ".join(new_parts)
            result.append(new_word)
        # If [var] is not found in the word, replace special characters in the entire word
        else:
            new_word = re.sub(pattern, " ", word)
            result.append(new_word)
    return " ".join(result)