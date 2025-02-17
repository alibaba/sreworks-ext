import os
import json

with open(os.environ.get('PYTHON_RUN_PATH')+"/completion", "r") as h:
    completion = h.read()

function = None
if os.path.exists(os.environ.get('PYTHON_RUN_PATH')+"/function"):
    with open(os.environ.get('PYTHON_RUN_PATH')+"/function", "r") as h:
        function = h.read()

try:
    context = onNext(ChainContext(message=completion, function=function))
    if context.do.overwriteMessage is not None:
        print("<overwriteMessage>" + context.do.overwriteMessage + "</overwriteMessage>")
    if context.do.message is not None:
        print("<message>" + context.do.message + "</message>")
    elif context.do.function is not None:
        print("<function>" + json.dumps(context.do.function.model_dump()) + "</function>")
    elif context.do.finalAnswer is not None:
        print("<finalAnswer>" + context.do.finalAnswer + "</finalAnswer>")
except Exception as e:
    print("<error>"+str(e)+"</error>")