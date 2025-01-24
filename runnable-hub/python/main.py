#!/usr/bin/env python3

from workers.processWorker.worker import Worker as ProcessWorker

from runnable import RunnableHub


if __name__ == "__main__":

    runnableHub = RunnableHub()
    runnableHub.registerWorker(ProcessWorker())
