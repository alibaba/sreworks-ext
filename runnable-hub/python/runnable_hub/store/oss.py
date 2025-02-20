import os
import oss2

from ..interface import RunnableFileStore


class RunnableOssFileStore(RunnableFileStore):
    def __init__(self, accessKey,  secretKey, endpoint, bucketName):
        auth = oss2.Auth(accessKey, secretKey)
        self.bucket = oss2.Bucket(auth, endpoint, bucketName)

    def saveFile(self, filePath, content):
        result = self.bucket.put_object(filePath, content)

        if result.status != 200:
            raise Exception(f"Failed to save file to OSS. Status: {result.status}")

    def readFile(self, filePath) -> str:
        try:
            object_stream = self.bucket.get_object(filePath)
            content = object_stream.read().decode('utf-8')
            return content
        except oss2.exceptions.NoSuchKey:
            raise FileNotFoundError(f"File {filePath} does not exist in OSS")
        except Exception as e:
            raise Exception(f"Failed to read file from OSS: {str(e)}")