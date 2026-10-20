"""Shared dashboard event streaming, independent of library features."""
import asyncio
from fastapi import WebSocketDisconnect


async def stream_events(websocket, service):
    await websocket.accept()
    queue, backlog = service.register_client()
    disconnected = asyncio.create_task(websocket.receive_text())
    next_event = None
    try:
        for event in backlog:
            await websocket.send_json(event)
        while True:
            next_event = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait([disconnected, next_event], return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                break
            await websocket.send_json(next_event.result())
    except WebSocketDisconnect:
        pass
    finally:
        pending = [task for task in (disconnected, next_event) if task]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        service.unregister_client(queue)
