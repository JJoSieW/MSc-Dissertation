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
""" AMI data preparation script
"""

import numpy as np
import os, sys, argparse
from os.path import join, basename, splitext, exists, dirname
import math
import matplotlib.pyplot as plt
import ami
import json


if __name__ == "__main__":
    
    # ================================
    # Parse input arguments
    # ================================
    #
    parser = argparse.ArgumentParser(description='Convert MIDI file into features.')
    parser.add_argument('--model', default='ami12', help='Model name (ami06, ami12, ami24)')
    parser.add_argument('--sample_freq', default=12, help='The number of samples per quarter note (default 12)', type=int)
    parser.add_argument('--key_shifts', default=7, help='number of keyshifts (default 7)', type=int)
    parser.add_argument('--n_beats_per_phrase', default=8, help='The number of beats per music phrase (default 8)', type=int)
    parser.add_argument('--midi_dir', default='data/midi/ami/classical', help='Directory of input midi files')
    parser.add_argument('--data_dir', default='data/features/ami/classical', help='Directory of output feature files')
    parser.add_argument('--model_dir', default='models', help='Directory of model')
    args = parser.parse_args()

    # ================================
    # Setup paths
    # ================================
    #
    if args.model == 'ami12':
        max_seq = 1024  # match emopia
        n_layer = 12 # match emopia
        n_head = 8  # match emopia
        n_embd = 768  # try different values, compare with emopia and analyse
    elif args.model == 'ami2':
        max_seq = 2048 
        n_layer = 16
        n_head = 16
        n_embd = 1024
    elif args.model == 'ami24':
        max_seq = 2400
        n_layer = 20
        n_head = 16
        n_embd = 1024
    else:
        sys.exit('Unknown model {}'.format(args.model))

    midi_dir = args.midi_dir
    data_dir = args.data_dir
    model_dir = join(args.model_dir, '{}-template'.format(args.model))
    if not exists(model_dir):
        os.makedirs(model_dir)

    # Save vocab and special tokens map
    tokenizer = ami.AmiTokenizer(sample_freq=args.sample_freq, max_len=max_seq)
    tokenizer.save_pretrained(model_dir)

    # Save config
    config = ami.AmiConfig(
        sample_freq=args.sample_freq,
        key_shifts=args.key_shifts,
        n_positions=max_seq,
        n_ctx=max_seq,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        max_timestep_len=args.sample_freq*args.n_beats_per_phrase,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_ids = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        vocab_size = tokenizer.vocab_size,
    )
    config.save_pretrained(model_dir)

    # Prepare training dataset
    dataset = ami.MusicDataset(
        config=config, 
        tokenizer=tokenizer, 
        feature_dir=data_dir, 
        block_size=config.n_ctx, 
        midi_dir=midi_dir)
    print(dataset.__repr__)
