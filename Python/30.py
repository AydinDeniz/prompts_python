
import asyncio
import websockets

# Dictionary to store connected clients
connected_clients = set()

async def chat_handler(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            # Broadcast message to all connected clients
            await asyncio.wait([client.send(message) for client in connected_clients if client != websocket])
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)

# Start WebSocket server
start_server = websockets.serve(chat_handler, "localhost", 8765)

if __name__ == "__main__":
    print("WebSocket server started on ws://localhost:8765")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
