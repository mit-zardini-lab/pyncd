import asyncio
import base64
from dataclasses import dataclass, field
import enum
import uuid
import websockets
from collections.abc import Mapping
from typing import TypedDict, Literal, Any, NotRequired
import json
import data_transfer.term_json as dtj
import data_structure.Term as fd
import random


type Websocket = Any

'''
The three parties, and what each does:

 - The server runs independently of the Jupyter kernel.
 - A data client, meaning a notebook, connects to the server over a
   WebSocket and sends it messages.
 - A diagram client, meaning a browser page running tsncd, connects to the
   server and receives the latest term the data client sent.

Display is one-way and capture is not. A `renderRequest` carries a term out to
the browser and an image comes back on a `renderResult`, correlated by
`requestId` because the two travel over different connections. The renderer
cannot run outside a browser, because the diagram's geometry exists only once
CSS has laid it out, so the round trip is how a notebook gets a picture at all.

The wire format is written up in `obsidian/05-backends/Diagram Wire Format.md`,
mirrored at `tsncd/PROTOCOL.md`. Keep the two in step.
'''

SERVER_HOST = 'localhost'
SERVER_PORT = 8765
SERVER_URI = f'ws://{SERVER_HOST}:{SERVER_PORT}'

'''
A captured PNG runs to several megabytes once base64'd, and `websockets`
defaults to rejecting a frame above 1 MiB, which drops the connection mid
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

class ColorMode(enum.Enum):
    DARK = 'dark'
    LIGHT = 'light'


class AxisHover(enum.Enum):
    '''Where an axis answers the pointer in a figure. Under `LEGEND`, the client's
    default, resting the pointer on the axis's legend row halos every wire of the
    axis and glows every name of it, and the wires and names answer no pointer.
    Under `EVERYWHERE` a wire or a name of the axis lights the same, and the
    legend row with it. Under `OFF` no halo is drawn and nothing answers.'''
    OFF = 'off'
    LEGEND = 'legend'
    EVERYWHERE = 'everywhere'


class RenderHandlerSettings(TypedDict, total=False):
    '''
    Display options forwarded verbatim to the TypeScript client. It mirrors
    `src/display/Render/RenderHandlerSettings.ts`.

    Partial by design, because the client merges whatever arrives over its own
    defaults, so omitting a key leaves that option at its default.
    '''
    darkMode: bool
    blockBackground: Literal['none', 'subtle', 'medium', 'strong']
    blockHoverIntensity: float
    debugBorders: bool
    coreDebug: bool
    # Wrap width in px, and so the diagram's aspect ratio: narrower means more
    # rows and a taller figure, wider means fewer rows and a flatter one.
    width: int
    # Whether `BlockOperator` bodies are drawn as sub-diagrams beside the main
    # figure (the client's default). Off, the figure is the high-level view
    # alone, so that each body can be rendered as its own figure.
    subBlocks: bool
    # The `BlockTag`s whose bodies have already been drawn, each as its
    # `uid._id`. The client leaves a body named here out and still draws the
    # box in the main figure. The sender keeps the record, because the client
    # wipes its render target on every message and a capture draws into a
    # second target that never saw the first.
    # `notebooks/display/remember_drawn_blocks.py` keeps it for a notebook.
    drawnBlockTags: list[int]
    # Whether a Grab and a Drop are labelled with the tape slot they reach.
    # On by default (the client's): the two are the same parameter but are
    # drawn in different rows with no line between them, so the label is what
    # pairs them. The label is the slot's name where it has one, which
    # `para.new_slot` assigns as 's0', 's1' and so on, and two hex digits of
    # its UID where it has none, with a hue derived from that UID.
    tapeLabels: bool
    # Whether a table of the term's axes, each with its size and its code name,
    # is drawn beside the figure. Off by default. The rows travel in the
    # `auxiliary` field of the message.
    legend: bool
    # Whether a block or an operator with a standard expansion opens an
    # inspection box when the pointer rests on it. Off by default. What a box
    # shows travels in the `auxiliary` field of the message.
    inspectionBoxes: bool
    # Where an axis answers the pointer: `legend`, the client's default, from
    # its legend row alone, `everywhere` from any wire or name of it as well,
    # and `off` nowhere. The value of an `AxisHover`.
    axisHover: Literal['off', 'legend', 'everywhere']
    # The name of what the page shows. The heading of the page and the name of
    # its tab read `tsncd - <title>`, and `tsncd` when no title is sent.
    title: str
    # The size, in em, of the label an axis carries on its wire. The client's
    # default is 0.8, and the layout measures the label at the size it is drawn.
    axisLabelFontSize: float


class AxisLegendRow(TypedDict):
    '''One row of the legend: the axis as latex and as text, the integer its
    size comes to or `None`, the code forms of its name and of its size, and the
    uid of every axis of the term the row stands for. tsncd links the row to the
    wires of the figure through the uids, so resting the pointer on the row
    halos those wires and resting it on one of them shades the row.'''
    latex: str
    text: str
    size: int | None
    codeName: str | None
    sizeCodeName: str | None
    uids: list[int]

class CodeReferenceRecord(TypedDict):
    '''A `cat.CodeReference` on the wire, with its url resolved. `icon` names the
    icon tsncd draws before the link, `huggingface` for a link into a repository
    on Hugging Face, or is `None`.'''
    label: str
    url: str | None
    path: str | None
    line: int | None
    endLine: int | None
    icon: str | None

class BlockInformation(TypedDict):
    '''What an inspection box shows for a block, beside the block's body. `formula`
    is LaTeX drawn under the title. A block whose aesthetics say
    `cat.BlockDrawing.BODY_IN_PLACE` is drawn in the figure as its body alone, and
    its box shows these fields and no body.'''
    title: str | None
    formula: str | None
    description: str | None
    references: list[CodeReferenceRecord]

class OperatorExpansion(TypedDict):
    '''What an inspection box shows for an operator with a standard expansion.
    `expansion` is the expanded morphism as its own exported term, and
    `auxiliary` is the auxiliary information of that morphism, so an operator
    inside the expansion can be opened in turn. `references` are the places in a
    codebase the operator stands for, listed under the description.'''
    operator: str
    latex: str | None
    formula: str
    description: str
    expansion: str
    auxiliary: 'DiagramAuxiliary'
    references: list[CodeReferenceRecord]

class DiagramAuxiliary(TypedDict, total=False):
    '''
    Information sent beside a term for the legend and the inspection boxes.
    It mirrors `src/advanced_display/AuxiliaryInformation.ts`. Every part is
    optional, and a message with no `auxiliary` field draws as it did before the
    field existed. `blocks` is keyed by the uid of each block's tag and
    `expansions` by the number tsncd's importer gives each `Broadcasted`, both
    as strings, which is how JSON keys an object.
    `websocket_transfer/auxiliary_information.py` assembles it.
    '''
    legend: list[AxisLegendRow]
    blocks: dict[str, BlockInformation]
    expansions: dict[str, OperatorExpansion]

class LocalisedExpansion(TypedDict, total=False):
    '''The description of an expansion under one localisation, where it differs
    from the exported one, and the localised descriptions of its nested auxiliary,
    where any differ.'''
    description: str
    auxiliary: 'LocalisedDescriptions'

class LocalisedDescriptions(TypedDict):
    '''The descriptions of one localisation that differ from the ones of the
    `DiagramAuxiliary` a page was exported with, keyed as that auxiliary keys them.
    `websocket_transfer/localise_descriptions.py` assembles it, and it mirrors
    `src/data_transfer/embedded_localisations.ts`.'''
    blocks: dict[str, str]
    expansions: dict[str, LocalisedExpansion]

class EmbeddedLocalisations(TypedDict):
    '''What a standalone page carries for its localisation toggle: the name of
    the localisation the page was exported with, and every localisation by name,
    in the order the toggle lists them. Only descriptions are localised, because a
    title sets the drawn size of its block.'''
    default: str
    localisations: dict[str, LocalisedDescriptions]

class DataUpdate(TypedDict):
    msgType: Literal['dataUpdate']
    data: dtj.JSONDataStructure
    settings: NotRequired[RenderHandlerSettings]
    auxiliary: NotRequired[DiagramAuxiliary]

class DataRequest(TypedDict):
    msgType: Literal['dataRequest']

class CaptureOptions(TypedDict, total=False):
    '''
    How the image should be cut. It mirrors `src/data_transfer/capture.ts`.

    Partial like `RenderHandlerSettings`, and for the same reason: an omitted
    key takes the client's default rather than whatever the previous send
    left behind.

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

    `disturbDisplay` settles whether that render lands on screen. Left out or
    true, it does, and the capture doubles as a send. False, and the browser
    draws into an off-screen target instead, leaving whatever is on screen -
    and the term the server holds for a reload, exactly as it was.
    '''
    msgType: Literal['renderRequest']
    requestId: str
    data: dtj.JSONDataStructure
    settings: NotRequired[RenderHandlerSettings]
    auxiliary: NotRequired[DiagramAuxiliary]
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
    auxiliary: DiagramAuxiliary | None = None
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

    async def send_to_diagrams(self, message: str) -> None:
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
                        json.dumps(self.held_data_update())
                    )
                return handlerInformation, {'msgType': 'Connected'}
            case {'msgType': 'dataUpdate', 'data': data}:
                # Mapping patterns match on a subset, so a client that sends no
                # settings still lands here, and receives the empty dict, and
                # the TypeScript side falls back to its own defaults.
                self.data_structure = data
                self.settings = msg.get('settings') or RenderHandlerSettings()
                self.auxiliary = msg.get('auxiliary')
                print('Data Updated.')
                await self.send_to_diagrams(json.dumps(self.held_data_update()))
                return handlerInformation, {'msgType': 'DataReceived'}
            case {'msgType': 'renderRequest', 'requestId': requestId, 'data': data}:
                disturb = msg.get('disturbDisplay', True)
                if disturb:
                    # Stored like a `dataUpdate`, so a browser that reloads
                    # after the capture comes back up on the same diagram.
                    self.data_structure = data
                    self.settings = msg.get('settings') or RenderHandlerSettings()
                    self.auxiliary = msg.get('auxiliary')
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
                # term, and only the first image back is used.
                await self.send_to_diagrams(json.dumps(with_auxiliary({
                    'msgType': 'renderRequest',
                    'requestId': requestId,
                    'data': data,
                    # The request's own settings rather than `self.settings`, because the
                    # latter is only kept current for renders that disturb.
                    'settings': msg.get('settings') or RenderHandlerSettings(),
                    'capture': msg.get('capture') or CaptureOptions(),
                    'disturbDisplay': disturb,
                }, msg.get('auxiliary'))))
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
                    return handlerInformation, self.held_data_update()
                else:
                    return handlerInformation, {'msgType': 'No Data Available'}
            case _:
                raise ValueError('Unknown message type: ' + str(msg))

    def held_data_update(self) -> DataUpdate:
        '''The term the server holds, as the `dataUpdate` a reconnecting page is
        sent, with the auxiliary information where the sender gave any.'''
        return with_auxiliary({
            'msgType': 'dataUpdate',
            'data': self.data_structure,
            'settings': self.settings,
        }, self.auxiliary)

    async def worker(self) -> None:
        while True:
            message = await self.message_queue.get()
            # for client in self.connected_clients:
            #     await client.send(message)

    async def main(self) -> None:
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
    # Set to request an image back, which turns the send into a `renderRequest`
    # and keeps the connection open until the reply lands.
    capture: CaptureOptions | None = None
    timeout: float = DEFAULT_CAPTURE_TIMEOUT
    disturb_display: bool = True
    # What the legend and the inspection boxes show, sent beside the term.
    # `None` leaves the field out of the message, and the page draws as it
    # did before the field existed.
    auxiliary: DiagramAuxiliary | None = None
    result: RenderResult | None = field(default=None, init=False)

    @classmethod
    async def template(
        cls,
        term: fd.GeneralTerm,
        settings: RenderHandlerSettings | None = None,
        capture: CaptureOptions | None = None,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
        disturb_display: bool = True,
        auxiliary: DiagramAuxiliary | None = None,
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
            disturb_display=disturb_display,
            auxiliary=auxiliary)
        await client.main()
        return client.result

    async def main(self) -> None:
        try:
            async with websockets.connect(
                    SERVER_URI, max_size=MAX_MESSAGE_BYTES) as websocket:
                await websocket.send(self.handshake)
                connected = await websocket.recv()
                print(f'Received from server: {connected}')
                if self.capture is None:
                    await websocket.send(json.dumps(with_auxiliary({
                        'msgType': 'dataUpdate',
                        'data': self.data,
                        'settings': self.settings
                    }, self.auxiliary)))
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
        reply cannot be identified as the next message, so the client reads until
    the matching
        `requestId` shows up. The whole wait is bounded, because a browser that
        is wedged would otherwise hang the notebook cell indefinitely.
        '''
        requestId = uuid.uuid4().hex
        await websocket.send(json.dumps(with_auxiliary({
            'msgType': 'renderRequest',
            'requestId': requestId,
            'data': self.data,
            'settings': self.settings,
            'capture': self.capture or CaptureOptions(),
            'disturbDisplay': self.disturb_display,
        }, self.auxiliary)))
        async with asyncio.timeout(self.timeout):
            while True:
                message = json.loads(await websocket.recv())
                if (message.get('msgType') == 'renderResult'
                        and message.get('requestId') == requestId):
                    return message

def with_auxiliary[M: Mapping[str, Any]](
    message: M, auxiliary: DiagramAuxiliary | None,
) -> M:
    '''`message` with the `auxiliary` field where there is any, and as it stands
    where there is none, so a message with nothing to show beside the term is
    the message an older page reads.'''
    if auxiliary is None:
        return message
    return {**message, 'auxiliary': auxiliary}  # type: ignore[return-value]

async def send_term(
    term: fd.GeneralTerm,
    settings: RenderHandlerSettings | None = None,
    auxiliary: DiagramAuxiliary | None = None,
) -> None:
    print('Sending term to server...')
    await DataClient.template(term, settings=settings, auxiliary=auxiliary)

async def capture_term(
    term: fd.GeneralTerm,
    settings: RenderHandlerSettings | None = None,
    capture: CaptureOptions | None = None,
    timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    disturb_display: bool = True,
    auxiliary: DiagramAuxiliary | None = None,
) -> bytes:
    '''
    Render `term` in the connected browser and return the image bytes it drew.

    With `disturb_display` the diagram also lands on screen. Without it the
    browser draws off-screen and the display is left alone.

    It requires a diagram page to be open either way, because that page is what does
    the rendering, and there is nowhere else it could happen.
    `websocket_transfer.headless` covers the case where there is no browser to
    hand.
    '''
    print('Requesting render from server...')
    result = await DataClient.template(
        term, settings=settings, capture=capture, timeout=timeout,
        disturb_display=disturb_display, auxiliary=auxiliary)
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