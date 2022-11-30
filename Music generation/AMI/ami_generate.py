#!/usr/bin/env python3
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ning Ma, Dec 2019
# n.ma@sheffield.ac.uk
# AMI - Artificial Musical Intelligence
""" AMI music generation script """

import argparse
import logging
import os
from os.path import join, exists, basename, splitext
import numpy as np
import datetime
import json
import torch
import fastmidi
from ami import (
    AmiLMHeadModel,
    AmiTokenizer,
    MusicLoader,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = (AmiLMHeadModel, AmiTokenizer)

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser(description='AMI generation script.')
    parser.add_argument("--repetition_penalty", type=float, default=1., 
        help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--temperature", type=float, default=1.2,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--p", type=float, default=.95)

    parser.add_argument('--prompt_beats', default=16, 
        help='Number of beats from the seed midi to be used. If None, the entire midi is used', type=int)
    
    #parser.add_argument("--model_name_or_path", default='models/ami24-electronic', type=str, help="Model path")
    #parser.add_argument('--prompt', default='["<BOS>", "keysig:0", "tempo:120"]', help='Prompt midi file, json token file, or token list string')
    #parser.add_argument('--prompt', default='outputs/ami2400-electronic/ami2400-electronic_200209-173934.json', help='Prompt midi file, json token file, or token list string')
    
    parser.add_argument("--model_name_or_path", default='models/ami24-classical', type=str, help="Model path")

    # Possible genres:
    #     "<Classical>", "<Rock>", "<Electronic>", "<Jazz>", "<Country>"
    # Possible classical styles:
    #     "<Bach>", "<Beethoven>", "<Brahms>", "<Chopin>", "<Handel>", "<Liszt>", "<Mendelssohn>", "<Mozart>",
    #     "<Rachmaninoff>", "<Scarlatti>",  "<Schubert>", "<Schumann>", "<Tchaikovsky>", "<Vivaldi>"
    # Possible Key Signatures: 
    #     "keysig:0" to "keysig:11" - C Major to B Major,
    #     "keysig:12" to "keysig:21" - C minor to B minor
    #  
    parser.add_argument('--prompt', default='["<Rock>", "<BOS>", "keysig:0", "tempo:96"]', help='Prompt midi file or token list')
    #parser.add_argument('--prompt', default='data/midi/examples/twinkle_twinkle.mid', help='Prompt midi file or token list')
    #parser.add_argument('--prompt', default='["<Classical>", "<Mozart>", "<BOS>"]', help='Prompt midi file or token list')
    #parser.add_argument('--prompt', default='["<Rock>", "<BOS>", "keysig:0", "tempo:120"]', help='Prompt midi file or token list')
    #parser.add_argument('--prompt', default='["<Electronic>", "<BOS>", "keysig:0", "tempo:120"]', help='Prompt midi file or token list')
    parser.add_argument('--style', default=None, help='style token list')
    #parser.add_argument('--style', default='["<Classical>", "<Vivaldi>"]', help='style token list')

    parser.add_argument('--instruments', default=None, #'[0]', # Piano
        help='Instruments to be used: None, or a list of General MIDI Program numbers 0-127 and 128 for Drumkit, e.g. "[0]" for Grand Piano') 
    
    parser.add_argument('--generate_beats', default=64, 
        help='The number of beats to be generated', type=int)

    parser.add_argument("--stop_token", type=str, default=fastmidi.TOKEN_EOS, help="Token at which text generation is stopped")
    parser.add_argument('--output_dir', default='outputs', help='Path to output midi files (default ./outputs/)')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    #set_seed(args)

    # Initialize the model and tokenizer
    model_class, tokenizer_class = MODEL_CLASSES
    
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.tokenizer = tokenizer
    model.to(args.device)

    logger.info(args)

    output_dir = join(args.output_dir, basename(args.model_name_or_path))
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Prepare prompt
    loader = MusicLoader(model.config, tokenizer)
    print('-'*80)
    print('Prompt: {}'.format(args.prompt))
    if args.prompt[0] == '[':
        tokens = json.loads(args.prompt)
        prompt_token_ids, prompt_token_timesteps = loader.load_prompt(tokens, n_beats=args.prompt_beats)
        args.prompt = basename(args.model_name_or_path)
        if fastmidi.TOKEN_BOS in tokens:
            for tok in tokens:
                if tok == fastmidi.TOKEN_BOS:
                    break
                else:
                    args.prompt += '_' + tok.replace('<','').replace('>','')
        # If instruments not set, use the default ones for the genre
        if args.instruments is None and tokens[0] in fastmidi.GENRE_TOKENS:
            args.instruments = fastmidi.GENRE_INSTRUMENTS[tokens[0]]

    else:
        prompt_token_ids, prompt_token_timesteps = loader.load_prompt(args.prompt, n_beats=args.prompt_beats, style=args.style)

    if args.instruments is None:
        filter_instrument_ids = None
    else:
        if type(args.instruments) is list:
            instruments = args.instruments
        else:
            instruments = json.loads(args.instruments)
        filter_instruments = ['{}:{}'.format(fastmidi.TOKEN_INSTRUMENT, i) for i in range(fastmidi.N_INSTRUMENTS) if i not in instruments]
        filter_instrument_ids = tokenizer.convert_tokens_to_ids(filter_instruments)

    prompt_token_ids = torch.tensor([prompt_token_ids])
    prompt_token_ids = prompt_token_ids.to(args.device)
    prompt_token_timesteps = torch.tensor([prompt_token_timesteps])
    prompt_token_timesteps = prompt_token_timesteps.to(args.device)

    output_sequences = model.generate(
        input_ids=prompt_token_ids,
        input_timesteps=prompt_token_timesteps,
        filter_instrument_ids=filter_instrument_ids,
        generate_beats=args.generate_beats,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
    )

    # Batch size == 1. to add more examples please use num_return_sequences > 1
    generated_sequence = output_sequences[0].tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(generated_sequence)
    # Write generated tokens
    output_file, _= splitext(join(output_dir, basename(args.prompt)))
    date_and_time = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    output_file += '_' + date_and_time

    with open(output_file + '.json', 'w') as f:
        json.dump(predicted_tokens, f)

    # Save as midi file
    fm = fastmidi.FastMidi()
    fm.decode_from_tokens(predicted_tokens, sample_freq=model.config.sample_freq)
    fm.write(output_file + '.mid')
    fastmidi.plot(output_file + '.mid', fig_size=(16,9), fig_file=output_file+'.png')
    print('Generated midi: {}.mid'.format(output_file))



if __name__ == "__main__":
    main()
