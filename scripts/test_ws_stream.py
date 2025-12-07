# /scripts/test_ws_stream.py
# Developer Note: Lightweight smoke test for the WebSocket streaming endpoint.

import asyncio
import json
import os
import sys

try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover
    print("Please install websockets for this smoke test: pip install websockets", file=sys.stderr)
    sys.exit(1)


async def main():
    uri = os.getenv("WS_SMOKE_URL", "ws://localhost:8000/ws/chat/debug-test")
    payload = {
        "type": "user_message",
        "messages": [{"role": "user", "content": "Hello from the test harness!"}],
    }

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(payload))
        while True:
            raw = await websocket.recv()
            data = json.loads(raw)
            print(data)
            if data.get("type") in {"done", "error"}:
                break


if __name__ == "__main__":
    asyncio.run(main())

