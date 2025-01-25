import os

from abc import ABC, abstractmethod

class RunnableStore(ABC):
    @abstractmethod
    def save(self, filePath, content):
        pass

    @abstractmethod
    def read(self, filePath) -> str:
        pass

class RunnableLocalStore(RunnableStore):
    def __init__(self, path):
        self.path = path

    def save(self, filePath, content):
        fullPath = os.path.join(self.path, filePath)
        print(fullPath)
        directory = os.path.dirname(fullPath)
        
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # 写入文件
        with open(fullPath, 'w') as file:
            file.write(content)

    def read(self, filePath) -> str:
        fullPath = os.path.join(self.path, filePath)
        
        # 检查文件是否存在
        if not os.path.exists(fullPath):
            raise FileNotFoundError(f"File {fullPath} does not exist")
        
        # 读取文件内容
        with open(fullPath, 'r') as file:
            return file.read()