'''Choosing how a tape member is labelled in a diagram or a listing.

Written by Claude Fable 5.1, effort 80.

A slot label is the slot's name, `W_{G}` for the gate weight, and inside a loop
that name stands for a different member of the tape on every iteration.
`INDEXED` writes the index of every loop that selects a member after the name,
`W_{G}[l]`, so a diagram of a repeated layer shows which member each grab and
drop touches. `CODE_FORM` writes the member's code form in typewriter, with the
same indices, `weight_G[layer]`. `NAMED` leaves the labels as the term names
them. A loop variable is carried across its loop and is indexed by the loops
outside it alone. `para.processing.tape_members` reads the members, and
`present` rebuilds the grabs and drops with the labels chosen. A notebook makes
the choice once in its `DiagramSettings`.
'''
from __future__ import annotations

import enum

import pandas

import agent_display.morphism_ir as morphism_ir
import data_structure.Term as fd
import para.processing.tape_members as tape_members


class TapeNaming(enum.Enum):
    NAMED = 'named'
    INDEXED = 'indexed'
    CODE_FORM = 'code_form'


def indexed_name(touch: tape_members.TapeTouch) -> fd.DynamicName | None:
    '''The slot's name with its indices written after it, and `None` for a
    member no loop indexes, whose label is right as it stands.'''
    if not touch.indices():
        return None
    return fd.DynamicName(body=touch.indexed_latex(), code_form=touch.indexed_code_form())


def code_form_name(touch: tape_members.TapeTouch) -> fd.DynamicName:
    '''The member's code form with its indices, drawn in typewriter.'''
    text = touch.indexed_code_form()
    return fd.DynamicName(
        body=text, settings=fd.DynamicNameSettings(typewriter=True), code_form=text)


def present[T](term: T, naming: TapeNaming) -> T:
    '''`term` with its tape members labelled as `naming` asks. `NAMED` returns
    it as it stands.'''
    match naming:
        case TapeNaming.NAMED:
            return term
        case TapeNaming.INDEXED:
            return tape_members.rename_tape_members(term, indexed_name)
        case TapeNaming.CODE_FORM:
            return tape_members.rename_tape_members(term, code_form_name)


def member_kind(member: tape_members.TapeMember) -> str:
    if member.carried_by is not None:
        return f'loop variable of {member.carried_by.code_form}'
    if member.indices:
        return 'one per iteration'
    return 'one'


def members_frame(members: fd.Prod[tape_members.TapeMember]) -> pandas.DataFrame:
    '''One row per member: its code form with its indices, how it is drawn,
    whether it is one member, one per iteration or a loop variable, the array
    it holds, how many grabs and drops touch it, and the blocks it is touched
    under.'''
    return pandas.DataFrame([{
        'member': member.indexed_code_form(),
        'drawn': member.indexed_latex(),
        'kind': member_kind(member),
        'array': morphism_ir.array_type(member.array()),
        'grabs': member.grabs(),
        'drops': member.drops(),
        'under': '; '.join(' > '.join(path) for path in member.block_paths()),
    } for member in members])
