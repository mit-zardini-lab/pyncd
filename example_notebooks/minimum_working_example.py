import os
import sys

# Puts the repository root on sys.path and makes it the working directory, so the
# imports and relative paths below resolve. See fix_notebook_dir.py.
import fix_notebook_dir

# The rendered diagrams use box-drawing characters, which the default Windows
# console code page cannot encode.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import construction_helpers as ch # Needed for @ auto-alignment
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import display as dpl
import websocket_transfer.websockets_transfer as wst

import subprocess
import asyncio

from typing import (
    Callable,
    Any,
    Literal
)

commands: dict[str, Callable[[], Any]] = {}

def attach_command(name: str):
    def name_wrapper(func):
        commands[name] = func
        return func
    return name_wrapper

#################
## CONVOLUTION ##
#################


def convolution_matrix():
    convolution_reindexing = cat.StrideMorphism.from_matrix(
        (1, 1),
        dom_names = ("x'", 'w'),
        cod_names=('x',),
        name='+'
    )
    c_in_axis = fd.DynamicName('c', fd.DynamicName('in')).capture(cat.RawAxis())
    convolution_matrix = (convolution_reindexing * c_in_axis) >> cat.Reals()
    return convolution_matrix

@attach_command('Convolution Matrix')
async def render_convolution():
    print('Convolution Matrix')
    _convolution_matrix = convolution_matrix()
    dpl.print_category(_convolution_matrix)
    await wst.send_term(_convolution_matrix)

def convolution_full():
    _convolution_matrix = convolution_matrix()
    c_out = fd.DynamicName('c', fd.DynamicName('out')).capture(cat.RawAxis())
    linear = ops.Linear.template(2, c_out)
    convolution_full = _convolution_matrix @ linear
    return convolution_full

@attach_command('Convolution Full')
async def render_convolution_full():
    print('Convolution Full')
    _convolution_full = convolution_full()
    dpl.print_category(_convolution_full)
    await wst.send_term(_convolution_full)

#################
## TRANSFORMER ##
#################

def attention_core():
    qk_matmul = ops.Einops.template('q h k, x h k -> h q x')
    softmax = ops.SoftMax.template()
    mask = ops.WeightedTriangularLower().template()
    sv_matmul = ops.Einops.template('h q x, x h k -> q h k')
    _attention_core = cat.Block.template(
        qk_matmul @ softmax @ mask @ sv_matmul,
        title='Attention Core',
        fill_color='#C5BEDF'
    )
    return _attention_core

@attach_command('Attention Core')
async def render_attention_core():
    print('Attention Core')
    _attention_core = attention_core()
    dpl.print_category(_attention_core)
    await wst.send_term(_attention_core)

def attention_layer():
    _attention_core = attention_core()
    Lq = ops.Linear.template(('m',), 2, 'q')
    Lk = ops.Linear.template(('m',), 2, 'k')
    Lv = ops.Linear.template(('m',), 2, 'v')
    Lo = ops.Linear.template(2, ('m',), 'o')
    _attention_layer = (Lq * Lk * Lv) @ _attention_core @ Lo
    return _attention_layer

@attach_command('Attention Layer')
async def render_attention_layer():
    print('Attention Layer')
    _attention_layer = attention_layer()
    dpl.print_category(_attention_layer)
    await wst.send_term(_attention_layer)

def res(target: cat.BroadcastedCategory):
    addition = ops.AdditionOp.template()
    norm = ops.Normalize.template()
    return cat.Block.template(
        (0,0) @ target @ ops.AdditionOp.template() @ ops.Normalize.template(),
        title='Add \\& Norm',
        fill_color='#F1F4C1'
    )

def ffn_layer():
    return cat.Block.template(
        ops.Linear.template(1, ('d_ff',), 'in')
        @ ops.Elementwise.template()
        @ ops.Linear.template(('d_ff',), 1, 'out'),
        title='Feed Forward',
        fill_color='#C1E8F7'
    )

@attach_command('FFN Layer')
async def render_ffn_layer():
    print('FFN Layer')
    _ffn_layer = ffn_layer()
    dpl.print_category(_ffn_layer)
    await wst.send_term(_ffn_layer)

def transformer_core():
    _attention_layer = attention_layer()
    _ffn_layer = ffn_layer()
    # The layer below is repeated, so it has to map its domain back to itself.
    # Attention takes three inputs - the query, key and value sources - so the
    # single incoming wire is fanned out to all three first.
    res_attention = res((0, 0, 0) @ _attention_layer)
    res_ffn = res(_ffn_layer)
    _transformer = cat.Block.template(
        res_attention @ res_ffn,
        title='Transformer Layer',
        fill_color='#F3F3F4',
        repetition=nm.Integer(6)
    )
    return _transformer

def transformer():
    vocab_size = fd.DynamicName('v', settings=fd.DynamicNameSettings(overline=True))
    embedding = cat.Block.template(
        ops.Embedding.template(vocab_size,),
        title='Embedding',
        fill_color='#FCE0E1')
    aggregator = cat.Block.template(
        ops.Linear.template(1, (vocab_size,)) @ ops.SoftMax.template(),
        title='Aggregator',
        fill_color='#DBDFEF'
    )
    attention_ffn_network = transformer_core()
    _transformer = embedding @ attention_ffn_network @ aggregator
    return _transformer

@attach_command('Transformer')
async def render_transformer():
    print('Transformer')
    _transformer = transformer()
    dpl.print_category(_transformer) # type: ignore
    await wst.send_term(_transformer)

def print_options():
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
            print('Invalid choice. Please enter a valid command number, or q to quit.')
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
