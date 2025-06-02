import os

from ..interface import RunnableQueue, RunnableQueueBus
import redis.asyncio as redis


class RunnableRedisQueue(RunnableQueue):
    def __init__(self, name, client):
        self.name = name
        self.client = client

    async def send(self, content: str):
        await self.client.rpush(self.name, content)

    async def receive(self):
        data = await self.client.blpop(self.name, timeout=0)
        return data[1].decode('utf-8')

class RunnableRedisQueueBus(RunnableQueueBus):
    def __init__(self, host: str, port: int, db:int):
        self.pool = redis.ConnectionPool(host=host, port=port, db=db)

    def register(self, name: str):
        return RunnableRedisQueue(name, redis.Redis(connection_pool=self.pool))

