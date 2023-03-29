from typing import List, Tuple, Dict
import collections
import random

import yaml

from remi_edit import REMI
from miditok import Vocabulary, Event
from miditoolkit import MidiFile
import numpy as np
from tqdm import tqdm

import torch


Token_List = List[int]


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    path: path to YAML configuration file
    return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    return cfg


def greedy_decode(file_path, model, tokenizer, test_loader, SOS_IDX, EOS_IDX, device):
    """
    Decode single example from test dataloader greedily
    """
    decoded_tokens = [SOS_IDX]

    src, tgt = next(iter(test_loader))
    src = src.to(device).cuda()

    input_tokens = torch.tensor([[SOS_IDX]]).cuda()
    with torch.no_grad():
        for i in tqdm(range(200)):

            logits = model(src, input_tokens)

            logits = logits[:, logits.shape[1]-1, :]
            top_token = torch.argmax(logits).item()

            if top_token == EOS_IDX:
                break

            decoded_tokens.append(top_token)
            input_tokens = torch.cat((input_tokens, torch.tensor([[top_token]]).cuda()), dim=1)

    # Dump to midi
    src_midi = tokenizer.tokens_to_midi(src.tolist(), [(0, False)])
    src_midi.dump('{:}_src.mid'.format(file_path))

    tgt_midi = tokenizer.tokens_to_midi(tgt.tolist(), [(0, False)])
    tgt_midi.dump('{:}_tgt.mid'.format(file_path))

    gen_midi = tokenizer.tokens_to_midi([decoded_tokens], [(0, False)])
    gen_midi.dump('{:}_gen.mid'.format(file_path))

    return decoded_tokens


def generate_tokens(
        tokenizer,
        midi_path: str,
        instructions: List[Tuple[int, str]]) -> List[Tuple[int, Token_List]]:
    """
    Reads a MIDI path and a list of instructions to be applied at each bar and returns a list of tokens representing the MIDI and the instructions at each bar.
    """
    midi = MidiFile(midi_path)

    all_instructions = set([instruction for (_, instruction) in instructions])
    """
    for instruction in all_instructions:
        if instruction not in tokenizer.vocab._event_to_token:
            tokenizer.vocab.add_event(Event("Instruction", instruction, time=0, desc="Instruct_"+instruction))"""

    instruct_timeline = generate_instructions_timeline(
        instruction_list=instructions, tokenizer=tokenizer)
    tokens = tokenizer(midi)
    new_tokens = insert_instructions(tokens, instruct_timeline,
                                     tokenizer.vocab)
    return new_tokens


def timestamp_to_bar_range(midi: MidiFile, section: Tuple[int, int]):
    """ TODO
    (mainly needed for user inputs rather than training data)
    Determine the bars a section covers, based on the timestamp pointers.
    """
    pass


def generate_instructions_timeline(
        instruction_list: List[Tuple[int, str]], tokenizer: REMI) -> Dict[int, List[str]]:
    """
    (Maybe add track pointer as well as bar pointer?)
    Takes a list of bar numbers and an instructions and combines into a dictionary of bar numbers and instruction events at that bar.
    """
    instruction_list = [
        (n,
         Event("Instruction",
               instruction,
               time=0,
               desc="An instruction indicator indicating playing under the " +
               instruction + " annotation."))
        for (n, instruction) in instruction_list
    ]

    timeline = collections.defaultdict(list)
    for a, b in instruction_list:
        if b.value != '':
            timeline[a].extend(tokenizer.events_to_tokens([b]))
    return dict(timeline)


def insert_instructions(tokens: Token_List,
                        instruction_timeline: Dict[int, List[str]],
                        vocab: Vocabulary) -> Token_List:
    """
    Iterate over our tokenised midi, when we encounter a bar token, we insert our instruction tokens based on the instruction_timeline dictionary
    """
    bar_count = -1
    for j, token in enumerate(tokens):
        if vocab.token_type(token) == "Bar":
            bar_count += 1  #move depending how we are indexing bars?
            if bar_count in instruction_timeline:
                for elem in instruction_timeline[bar_count]:
                    tokens.insert(j+1, elem)
                    j += 1
    return tokens


def remove_meta_tokens(token_sequence: Token_List,
                       vocab: Vocabulary) -> Token_List:
    """
    Iterates over token sequence removing any instruction tokens from the sequence to clean for MIDI reconstruction
    """
    for j, token in enumerate(token_sequence):
        if vocab.token_type(token) == "Instruction":
            token_sequence.pop(j)
    return token_sequence


def tokens_to_midi(tokenizer: REMI, token_sequence: Token_List) -> MidiFile:
    """
    Takes tokens and parses back into a MIDI object.
    """
    return tokenizer.tokens_to_midi([token_sequence])
