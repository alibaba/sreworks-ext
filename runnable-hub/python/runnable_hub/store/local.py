import os

from ..interface import RunnableFileStore


class RunnableLocalFileStore(RunnableFileStore):
    def __init__(self, path):
        self.path = path

    def saveFile(self, filePath, content):
        fullPath = os.path.join(self.path, filePath)
        directory = os.path.dirname(fullPath)

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 写入文件
        with open(fullPath, 'w') as file:
            file.write(content)

    def readFile(self, filePath) -> str:
        fullPath = os.path.join(self.path, filePath)

        # 检查文件是否存在
        if not os.path.exists(fullPath):
            raise FileNotFoundError(f"File {fullPath} does not exist")

        # 读取文件内容
        with open(fullPath, 'r') as file:
            return file.read()
