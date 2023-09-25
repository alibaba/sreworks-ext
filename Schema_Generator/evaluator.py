import asyncio

class Evaluator:
    def __init__(self):
        self.results = []

    async def evaluate_json_program(self, program):
        for expr in program["@steps"]:
            self.results.append(await self.evaluate(expr))
        return self.results[-1] if self.results else None

    async def evaluate(self, expr):
        if isinstance(expr, dict):
            return await self.evaluate_object(expr)
        else:
            return expr

    async def evaluate_object(self, obj):
        if "@ref" in obj:
            index = obj["@ref"]
            if isinstance(index, int) and index < len(self.results):
                return self.results[index]
        elif isinstance(obj, list):
            return await self.evaluate_array(obj)
        else:
            values = await asyncio.gather(*[self.evaluate(e) for e in obj.values()])
            return {k: v for k, v in zip(obj.keys(), values)}

    async def evaluate_array(self, array):
        return await asyncio.gather(*[self.evaluate(e) for e in array])
