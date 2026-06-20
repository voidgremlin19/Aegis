import asyncio
from server import load_resources
import server
import json

load_resources("gpt2")
print("Resources loaded")
async def test():
    for chunk in server.wrapper.generate_stream(prompt="Hello", max_new_tokens=5):
        print(chunk)
asyncio.run(test())
