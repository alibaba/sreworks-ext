import os
from typing import Dict, List
from abc import ABC, abstractmethod


class RunnableFileStore(ABC):
    @abstractmethod
    def saveFile(self, filePath, content):
        pass

    @abstractmethod
    def readFile(self, filePath) -> str:
        pass


class RunnableDatabaseStore(ABC):
    @abstractmethod
    def queryRows(self, table: str, where: str) -> List[Dict]:
        pass

    @abstractmethod
    def insertRows(self, table: str, data: List[Dict]) -> str:
        pass


class RunnableQueueBus(ABC):

    @abstractmethod
    def register(self, name: str):
        pass

class RunnableQueue(ABC):
    @abstractmethod
    async def send(self, content: str):
        pass

    @abstractmethod
    async def receive(self):
        pass