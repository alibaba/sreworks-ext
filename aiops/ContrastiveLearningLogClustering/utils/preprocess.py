import re
import time
import pandas as pd
from collections import OrderedDict

def add_blank_token(log):
    new_log = re.sub('\s+','[blank]',log)
    return new_log

def add_var_token(rex, line):
    for currentRex in rex:
        line = re.sub(currentRex, '[var]', line)
    line = re.sub('(\[var\])+','[var]',line)
    # line = re.sub('(\s*\[var\]\s*)+',' [var] ',line)
    return line

def clean_log(log):
    log = replace_special_characters(log)

    pattern = r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
    log = re.sub(pattern, ' ', log)

    log = re.sub("\s{2,}"," ",log)
    return log

def replace_special_characters(s):
    words = s.split()
    pattern = r"[^A-Za-z0-9]"
    result = []
    for word in words:
        if "[var]" in word:
            parts = word.split("[var]")
            new_parts = [re.sub(pattern, " ", part) for part in parts]
            # print(new_parts)
            new_word = " [var] ".join(new_parts)
            result.append(new_word)
        else:
            new_word = re.sub(pattern, " ", word)
            result.append(new_word)
    return " ".join(result)