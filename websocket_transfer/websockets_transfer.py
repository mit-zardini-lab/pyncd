import asyncio
import base64
from dataclasses import dataclass, field
import uuid
import websockets
from typing import TypedDict, Literal, Any, NotRequired
import json
import data_transfer.json as dtj
import data_structure.Term as fd
import random


type Websocket = Any

'''
The design is as follows:
 - We have a SERVER. The SERVER runs independently from Jupyter Notebook.
 - A CLIENT connects to the SERVER via WebSocket, and can send messages to it.
 - TypeScript Browser connects to the SERVER. It then receives the latest term from the CLIENT.

Display is one-way, but capture is not: a `renderRequest` carries a term out to
the browser and an image comes back on a `renderResult`, correlated by
`requestId` because the two travel over different connections. The renderer
cannot run outside a browser - the diagram's geometry only exists once CSS has
laid it out - so this round trip is how a notebook gets a picture at all.

The wire format is written up in `tsncd`'s `PROTOCOL.md`; keep the two in step.
'''

SERVER_HOST = 'localhost'
SERVER_PORT = 8765
SERVER_URI = f'ws://{SERVER_HOST}:{SERVER_PORT}'

'''
A captured PNG runs to several megabytes once base64'd, and `websockets`
defaults to rejecting frames above 1 MiB - which drops the connection mid
capture instead of reporting anything useful. Both ends have to raise it.
'''
MAX_MESSAGE_BYTES = 64 * 2**20

DEFAULT_CAPTURE_TIMEOUT = 60.0

type Message = (HandshakeMessage | DataUpdate | DataRequest
                | RenderRequest | RenderResult | GenericMessage)

class HandshakeMessage(TypedDict):
    msgType: Literal['identify']
    clientType: Literal['DiagramClient', 'DataClient']
    clientVersion: str
    clientID: str

class RenderHandlerSettings(TypedDict, total=False):
    '''
    Display options forwarded verbatim to the TypeScript client; mirrors
    `src/display/Render/RenderHandlerSettings.ts`.

    Partial by design - the client merges whatever arrives over its own
    defaults, so omitting a key leaves that option at its default.
    '''
    darkMode: bool
    debugBorders: bool
    coreDebug: bool
    # Wrap width in px, and so the diagram's aspect ratio: narrower means more
    # rows and a taller figure, wider means fewer rows and a flatter one.
    width: int

class DataUpdate(TypedDict):
    msgType: Literal['dataUpdate']
    data: dtj.JSONDataStructure
    settings: NotRequired[RenderHandlerSettings]

class DataRequest(TypedDict):
    msgType: Literal['dataRequest']

class CaptureOptions(TypedDict, total=False):
    '''
    How the image should be cut; mirrors `src/data_transfer/capture.ts`.

    Partial like `RenderHandlerSettings`, and for the same reason - an omitted
    key takes the client's default rather than whatever was asked for last.

    `padding` is not cosmetic. `HTMLDrawHandler` places its SVG layers at
    (-10, -10) relative to the diagram container, so a capture with no margin
    slices the overlay off two sides.
    '''
    format: Literal['png', 'svg']
    scale: float
    padding: int
    background: str | None

class RenderRequest(TypedDict):
    '''
    A `dataUpdate` whose sender is waiting for an image of the result.

    `disturbDisplay` decides whether that render lands on screen. Left out or
    true, it does, and the capture doubles as a send. False, and the browser
    draws into an off-screen target instead, leaving whatever is on screen -
    and the term the server holds for reloads - exactly as it was.
    '''
    msgType: Literal['renderRequest']
    requestId: str
    data: dtj.JSONDataStructure
    settings: NotRequired[RenderHandlerSettings]
    capture: NotRequired[CaptureOptions]
    disturbDisplay: NotRequired[bool]

class RenderResult(TypedDict):
    '''
    The reply, carrying either an image or the reason there isn't one. The
    server matches it to a waiting sender by `requestId` and forwards it
    verbatim.
    '''
    msgType: Literal['renderResult']
    requestId: str
    mime: NotRequired[str]
    payload: NotRequired[str]
    encoding: NotRequired[Literal['base64', 'utf-8']]
    width: NotRequired[float]
    height: NotRequired[float]
    error: NotRequired[str]

class GenericMessage(TypedDict):
    msgType: str

class CaptureError(RuntimeError):
    '''Raised when a capture round trip cannot produce an image.'''

@dataclass
class HandlerInformation:
    socket: Websocket
    clientType: Literal['DiagramClient', 'DataClient'] | None = None

@dataclass
class DataServer:
    data_structure: dtj.JSONDataStructure | None = None
    settings: RenderHandlerSettings = field(default_factory=RenderHandlerSettings)
    diagram_clients: dict[str, Websocket] = field(default_factory=dict)
    data_clients: dict[str, Websocket] = field(default_factory=dict)
    message_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    connected_clients: dict[int, HandlerInformation] = field(default_factory=dict)
    # requestId -> the socket waiting for that image.
    pending_captures: dict[str, Websocket] = field(default_factory=dict)

    async def handler(self, websocket):
        print('Client connected.')

        randomKey = random.randint(0, 2**12)
        handlerInformation = HandlerInformation(socket=websocket)
        self.connected_clients[randomKey] = handlerInformation

        try:
            async for message in websocket:
                handlerInformation, response = await self.process_message(
                    json.loads(message), 
                    handlerInformation)
                self.connected_clients[randomKey] = handlerInformation
                await self.send_to_one(websocket, json.dumps(response))
        finally:
            del self.connected_clients[randomKey]
            # A sender that walked away mid-capture would otherwise leave its
            # requestId behind, and the image would later be pushed at a closed
            # socket.
            for requestId in [
                key for key, socket in self.pending_captures.items()
                if socket is websocket
            ]:
                del self.pending_captures[requestId]
            print('Client disconnected.')

    async def send_to_one(self, client, message: str):
        await client.send(message)

    def diagram_sockets(self) -> list[Websocket]:
        return [
            client.socket
            for client in self.connected_clients.values()
            if client.clientType == 'DiagramClient'
        ]

    async def send_to_diagrams(self, message: str):
        for socket in self.diagram_sockets():
            await socket.send(message)

    async def process_message(self, 
            msg: Message,
            handlerInformation: HandlerInformation
        ) -> tuple[HandlerInformation, Message]:
        match msg:
            case {'msgType': 'identify', 'clientType': clientType, 'clientVersion': clientVersion, 'clientID': clientID}:
                print(f"Client identified: {clientType} v{clientVersion} (ID: {clientID})")
                handlerInformation.clientType = clientType
                if clientType == 'DiagramClient' and self.data_structure is not None:
                    print('Sending data.')
                    await self.send_to_one(
                        handlerInformation.socket,
                        json.dumps({
                            'msgType': 'dataUpdate',
                            'data': self.data_structure,
                            'settings': self.settings
                        })
                    )
                return handlerInformation, {'msgType': 'Connected'}
            case {'msgType': 'dataUpdate', 'data': data}:
                # Mapping patterns match on a subset, so a client that sends no
                # settings still lands here - it just gets the empty dict, and
                # the TypeScript side falls back to its own defaults.
                self.data_structure = data
                self.settings = msg.get('settings') or RenderHandlerSettings()
                print('Data Updated.')
                await self.send_to_diagrams(
                    json.dumps({
                        'msgType': 'dataUpdate',
                        'data': data,
                        'settings': self.settings
                    })
                )
                return handlerInformation, {'msgType': 'DataReceived'}
            case {'msgType': 'renderRequest', 'requestId': requestId, 'data': data}:
                disturb = msg.get('disturbDisplay', True)
                if disturb:
                    # Stored like a `dataUpdate`, so a browser that reloads
                    # after the capture comes back up on the same diagram.
                    self.data_structure = data
                    self.settings = msg.get('settings') or RenderHandlerSettings()
                else:
                    # Deliberately not stored. Overwriting here would leave the
                    # display intact only until the next reload, which is a
                    # disturbance with a delay on it.
                    print('Render request will not disturb the display.')
                if not self.diagram_sockets():
                    print(f'Render request {requestId} refused: no diagram client.')
                    return handlerInformation, {
                        'msgType': 'renderResult',
                        'requestId': requestId,
                        'error': (
                            'No DiagramClient is connected. Open the tsncd page '
                            '(npm run dev), or capture headlessly with '
                            'websocket_transfer.headless.'
                        ),
                    }
                print(f'Render requested: {requestId}.')
                self.pending_captures[requestId] = handlerInformation.socket
                # Every diagram client renders, so they all stay on the same
                # term; only the first image back is used.
                await self.send_to_diagrams(json.dumps({
                    'msgType': 'renderRequest',
                    'requestId': requestId,
                    'data': data,
                    # The request's own settings, not `self.settings` - the
                    # latter is only kept current for renders that disturb.
                    'settings': msg.get('settings') or RenderHandlerSettings(),
                    'capture': msg.get('capture') or CaptureOptions(),
                    'disturbDisplay': disturb,
                }))
                return handlerInformation, {
                    'msgType': 'RenderRequested', 'requestId': requestId}
            case {'msgType': 'renderResult', 'requestId': requestId}:
                requester = self.pending_captures.pop(requestId, None)
                if requester is None:
                    # A second diagram client answering a request the first one
                    # already won, or a reply that arrived after the sender gave
                    # up waiting. Either way there is nobody left to hand it to.
                    print(f'Render result {requestId} dropped: nobody waiting.')
                    return handlerInformation, {'msgType': 'RenderResultDropped'}
                print(f'Render result {requestId} forwarded.')
                await self.send_to_one(requester, json.dumps(msg))
                return handlerInformation, {'msgType': 'RenderResultForwarded'}
            case {'msgType': 'dataRequest'}:
                print('Data Requested.')
                if self.data_structure is not None:
                    return handlerInformation, {
                        'msgType': 'dataUpdate',
                        'data': self.data_structure,
                        'settings': self.settings
                    }
                else:
                    return handlerInformation, {'msgType': 'No Data Available'}
            case _:
                raise ValueError('Unknown message type: ' + str(msg))

    async def worker(self):
        while True:
            message = await self.message_queue.get()
            # for client in self.connected_clients:
            #     await client.send(message)

    async def main(self):
        async with websockets.serve(
                self.handler, SERVER_HOST, SERVER_PORT,
                max_size=MAX_MESSAGE_BYTES):
            print(f"Server started at {SERVER_URI}")
            await self.worker()

@dataclass
class DataClient:
    handshake: str
    data: str
    settings: RenderHandlerSettings = field(default_factory=RenderHandlerSettings)
    # Set to ask for an image back, which turns the send into a `renderRequest`
    # and keeps the connection open until the reply lands.
    capture: CaptureOptions | None = None
    timeout: float = DEFAULT_CAPTURE_TIMEOUT
    disturb_display: bool = True
    result: RenderResult | None = field(default=None, init=False)

    @classmethod
    async def template(
        cls,
        term: fd.GeneralTerm,
        settings: RenderHandlerSettings | None = None,
        capture: CaptureOptions | None = None,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
        disturb_display: bool = True,
    ) -> RenderResult | None:
        handshake: HandshakeMessage = {
            'msgType': 'identify',
            'clientType': 'DataClient',
            'clientVersion': '0.1.0',
            'clientID': 'unique-client-id-1234'
        }
        data = dtj.TermJSONConverter.export_to_json(term)
        client = cls(
            handshake=json.dumps(handshake),
            data=data,
            settings=settings if settings is not None else RenderHandlerSettings(),
            capture=capture,
            timeout=timeout,
            disturb_display=disturb_display)
        await client.main()
        return client.result

    async def main(self):
        try:
            async with websockets.connect(
                    SERVER_URI, max_size=MAX_MESSAGE_BYTES) as websocket:
                await websocket.send(self.handshake)
                connected = await websocket.recv()
                print(f'Received from server: {connected}')
                if self.capture is None:
                    await websocket.send(json.dumps({
                        'msgType': 'dataUpdate',
                        'data': self.data,
                        'settings': self.settings
                    }))
                    response = await websocket.recv()
                    print(f"Received from server: {response}")
                else:
                    self.result = await self.request_render(websocket)
        except Exception as e:
            if self.capture is not None:
                # Unlike a display, a capture has a return value the caller is
                # about to use, so a swallowed failure would surface later as a
                # confusing `None`.
                raise CaptureError(
                    'Capture failed. Check that `python run_server.py` is '
                    f'running and a tsncd page is open. Cause: {e!r}') from e
            print(f"Be sure to execute `python run_server.py` before running this client. An error occurred: {e}")

    async def request_render(self, websocket) -> RenderResult:
        '''
        Send the term and wait for the image of it.

        The server acknowledges the request before the browser has drawn
        anything, and the acknowledgement arrives on this same socket, so the
        reply cannot simply be the next message - we read until the matching
        `requestId` shows up. The whole wait is bounded, because a browser that
        is wedged would otherwise hang the notebook cell indefinitely.
        '''
        requestId = uuid.uuid4().hex
        await websocket.send(json.dumps({
            'msgType': 'renderRequest',
            'requestId': requestId,
            'data': self.data,
            'settings': self.settings,
            'capture': self.capture or CaptureOptions(),
            'disturbDisplay': self.disturb_display,
        }))
        async with asyncio.timeout(self.timeout):
            while True:
                message = json.loads(await websocket.recv())
                if (message.get('msgType') == 'renderResult'
                        and message.get('requestId') == requestId):
                    return message

async def send_term(
    term: fd.GeneralTerm,
    settings: RenderHandlerSettings | None = None,
):
    print('Sending term to server...')
    await DataClient.template(term, settings=settings)

async def capture_term(
    term: fd.GeneralTerm,
    settings: RenderHandlerSettings | None = None,
    capture: CaptureOptions | None = None,
    timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    disturb_display: bool = True,
) -> bytes:
    '''
    Render `term` in the connected browser and return the image bytes it drew.

    With `disturb_display` the diagram also lands on screen; without it the
    browser draws off-screen and the display is left alone.

    Requires a diagram page to be open either way - it is that page which does
    the rendering, and there is nowhere else it could happen.
    `websocket_transfer.headless` covers the case where there is no browser to
    hand.
    '''
    print('Requesting render from server...')
    result = await DataClient.template(
        term, settings=settings, capture=capture, timeout=timeout,
        disturb_display=disturb_display)
    if result is None:
        raise CaptureError('No render result was returned.')
    return result_to_bytes(result)

def result_to_bytes(result: RenderResult) -> bytes:
    '''Unwrap a `renderResult`, raising whatever error it carries instead.'''
    if error := result.get('error'):
        raise CaptureError(error)
    payload = result.get('payload')
    if payload is None:
        raise CaptureError(f'Render result carried no image: {result}')
    if result.get('encoding') == 'base64':
        return base64.b64decode(payload)
    return payload.encode('utf-8')

# asyncio.run(main())
# print('end.')

if __name__ == '__main__':
    server = DataServer()
    asyncio.run(server.main())