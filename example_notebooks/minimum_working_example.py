'''Sending the example expressions to an open tsncd page, one at a time.

Run it from the repository root, with a tsncd page open in a browser:

    python example_notebooks/minimum_working_example.py

It runs `run_server.py` as a subprocess, which holds the relay for as long as the
menu is open. The menu numbers the expressions built by the notebooks in this
folder, and the one that is chosen is sent to the page. Each expression is also
printed with `display.print_category`, which draws it in the terminal and needs
no browser.

The notebooks draw through `notebooks/display/notebook_diagrams.py` instead,
which captures the figure from a headless browser and embeds it in the cell. This
script is the other route: the relay on port 8765 and a page that a person is
looking at.

Written by Claude Opus 5 (1M context) at reasoning effort high, 2026-09-20.
'''
import sys; sys.path[:0] = ['.', 'notebooks']
import fix_notebook_dir

import asyncio
import os
import subprocess
from typing import Any, Callable, Literal

import construction_helpers as ch  # noqa: F401 - operator overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import display as dpl
import websocket_transfer.websockets_transfer as wst

# `print_category` draws with box-drawing characters, which the default Windows
# console code page cannot encode.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

commands: dict[str, Callable[[], Any]] = {}


def attach_command(name: str):
    def name_wrapper(func):
        commands[name] = func
        return func
    return name_wrapper


async def print_and_send(term: cat.Morphism, name: str) -> None:
    print(name)
    dpl.print_category(term)
    await wst.send_term(term)


def convolution_matrix() -> cat.Morphism:
    convolution_reindexing = cat.StrideMorphism.from_matrix(
        (1, 1),
        dom_names=("x'", 'w'),
        cod_names=('x',),
        name='+'
    )
    input_channels = fd.DynamicName('c', fd.DynamicName('in')).capture(cat.RawAxis())
    return (convolution_reindexing * input_channels) >> cat.Reals()


@attach_command('Convolution Matrix')
async def send_convolution_matrix() -> None:
    await print_and_send(convolution_matrix(), 'Convolution Matrix')


def convolution() -> cat.Morphism:
    output_channels = fd.DynamicName('c', fd.DynamicName('out')).capture(cat.RawAxis())
    return convolution_matrix() @ ops.Linear.template(2, output_channels)


@attach_command('Convolution')
async def send_convolution() -> None:
    await print_and_send(convolution(), 'Convolution')


def attention_core() -> cat.Morphism:
    return cat.Block.template(
        ops.Einops.template('q h k, x h k -> h q x')
        @ ops.SoftMax.template()
        @ ops.WeightedTriangularLower.template()
        @ ops.Einops.template('h q x, x h k -> q h k'),
        title='Attention Core',
        fill_color='#C5BEDF'
    )


@attach_command('Attention Core')
async def send_attention_core() -> None:
    await print_and_send(attention_core(), 'Attention Core')


def attention_layer() -> cat.Morphism:
    query = ops.Linear.template(('m',), 2, 'q')
    key = ops.Linear.template(('m',), 2, 'k')
    value = ops.Linear.template(('m',), 2, 'v')
    output = ops.Linear.template(2, ('m',), 'o')
    return (query * key * value) @ attention_core() @ output


@attach_command('Attention Layer')
async def send_attention_layer() -> None:
    await print_and_send(attention_layer(), 'Attention Layer')


def residual(target: cat.BroadcastedCategory) -> cat.Morphism:
    return cat.Block.template(
        (0, 0) @ target @ ops.AdditionOp.template() @ ops.Normalize.template(),
        title='Add \\& Norm',
        fill_color='#F1F4C1'
    )


def feed_forward() -> cat.Morphism:
    return cat.Block.template(
        ops.Linear.template(1, ('d_ff',), 'in')
        @ ops.Elementwise.template()
        @ ops.Linear.template(('d_ff',), 1, 'out'),
        title='Feed Forward',
        fill_color='#C1E8F7'
    )


@attach_command('Feed Forward')
async def send_feed_forward() -> None:
    await print_and_send(feed_forward(), 'Feed Forward')


def transformer_layers() -> cat.Morphism:
    # Attention reads the query, the key and the value, so the copied wire is
    # fanned out to three in front of it.
    return cat.Block.template(
        residual((0, 0, 0) @ attention_layer()) @ residual(feed_forward()),
        title='Transformer Layer',
        fill_color='#F3F3F4',
        repetition=nm.Integer(6)
    )


def transformer() -> cat.Morphism:
    vocabulary = fd.DynamicName('v', settings=fd.DynamicNameSettings(overline=True))
    embedding = cat.Block.template(
        ops.Embedding.template(vocabulary),
        title='Embedding',
        fill_color='#FCE0E1')
    aggregator = cat.Block.template(
        ops.Linear.template(1, (vocabulary,)) @ ops.SoftMax.template(),
        title='Aggregator',
        fill_color='#DBDFEF'
    )
    return embedding @ transformer_layers() @ aggregator


@attach_command('Transformer')
async def send_transformer() -> None:
    await print_and_send(transformer(), 'Transformer')


def print_options() -> None:
    print('Available commands:')
    for i, command in enumerate(commands):
        print(f'({i}) {command}')
    print('(q) Quit')


async def ask_input() -> None | Literal['Quit']:
    while True:
        print_options()
        choice = input('Enter command number, or q to quit: ')
        if choice.lower() == 'q':
            return 'Quit'
        try:
            command_name = list(commands.keys())[int(choice)]
        except (ValueError, IndexError):
            print(f'{choice!r} is not one of 0 to {len(commands) - 1} or q.')
            continue
        await commands[command_name]()


if __name__ == '__main__':
    server = subprocess.Popen(
        [sys.executable,
         os.path.join(fix_notebook_dir.REPO_ROOT, 'run_server.py')])
    print('Server started.')
    while True:
        command = asyncio.run(ask_input())
        if command == 'Quit':
            print('Exiting.')
            server.kill()
            break
