from runnable import RunnableWorker


class Worker(RunnableWorker):

    def __init__(self):
        print("processWorker init")


    def onNext(self):
        pass


