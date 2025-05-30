import asyncio

import aiomysql
from typing import Dict
from runnable_hub.interface import RunnableDatabaseStore


class RunnableMySQLStore(RunnableDatabaseStore):

    def __init__(self, host, port, user, password, db):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                autocommit=True
            )
        return self._pool

    async def queryRows(self, table, where:Dict):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                where_clause = ' AND '.join([f"{k}=%s" for k in where.keys()])
                sql = f"SELECT * FROM {table} WHERE {where_clause}"
                await cur.execute(sql, tuple(where.values()))
                result = await cur.fetchall()
                return result

    async def insertRows(self, table, data:Dict):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['%s'] * len(data))
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                await cur.execute(sql, tuple(data.values()))
                await conn.commit()
