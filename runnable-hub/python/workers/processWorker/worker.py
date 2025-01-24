from runnable import RunnableWorker, RunnableContext
from request import Request


class Worker(RunnableWorker):

    runnableCode = "PROCESS_WORKER"

    def __init__(self):
        print("processWorker init")


    def onNext(self, context: RunnableContext) -> RunnableContext:

        return context


