# /scripts/test_log_writer.py
# Developer Note: Validates the background log writer without touching the WebSocket path.

import asyncio
import uuid

from main import _background_log_writer, serialize_conversation_for_log  # type: ignore


async def main():
    queue: asyncio.Queue = asyncio.Queue()
    writer = asyncio.create_task(_background_log_writer(queue))

    conversation = serialize_conversation_for_log(
        [
            {"role": "user", "content": "Ping?"},
            {"role": "assistant", "content": "Pong!"},
        ]
    )

    filename = f"test_log_{uuid.uuid4().hex}.json"
    job = {"conversation": conversation, "meta": {"source": "log-writer-test"}, "filename": filename}
    await queue.put(job)
    await queue.put(None)

    await queue.join()
    await writer
    print(f"Wrote conversations/{filename}")


if __name__ == "__main__":
    asyncio.run(main())

