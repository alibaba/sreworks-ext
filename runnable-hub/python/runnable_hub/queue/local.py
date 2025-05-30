import os

from ..interface import RunnableQueueBus, RunnableQueue
import asyncio


class RunnableLocalQueue(RunnableQueue):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def send(self, content: str):
        await self.queue.put(content)

    async def receive(self):
        return await self.queue.get()


class RunnableLocalQueueBus(RunnableQueueBus):
    def __init__(self):
        pass

    def register(self, name: str):
        return RunnableLocalQueue(asyncio.Queue())
