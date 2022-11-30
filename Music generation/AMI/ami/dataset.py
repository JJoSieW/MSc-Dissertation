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
"""AMI Dataset"""

import numpy as np
import os, sys
from os.path import join, exists, basename, splitext
from progress.bar import Bar
import fastmidi
import pickle, json
import random
import torch
from torch.utils.data import Dataset


def list_all_files(root_dir, file_ext):
    '''Lists all files in a folder and its subfolders that match file_ext
    Args:
        root_dir: top directory
        file_ext: file extensions, e.g. ('.mid','.wav') or '.mid'
    Returns:
        A list of files in root_dir
    '''
    return sorted([join(root, name) for root, dirs, files in os.walk(root_dir) for name in files if name.endswith(file_ext)])


def get_dataset_name(config):
    name = 'seq{}_fs{}_ks{}'.format(config.n_ctx, config.sample_freq, len(config.key_shifts))
    return name

def preprocess_midi(midi_indir, midi_outdir, sample_freq=fastmidi.SAMPLE_FREQ):
    ''' Preprocess midi files to remove invalid or short ones
    '''
    midi_file_list = list_all_files(midi_indir, ('.mid','.midi','.MID','.MIDI'))
    n_files = len(midi_file_list)
    if n_files == 0:
        sys.exit('No midi files found in {}'.format(midi_indir))
    else:
        print('Found {} midi files'.format(n_files))

    if not exists(midi_outdir):
        os.makedirs(midi_outdir)

    cnt = 1
    for i in Bar('Processing Midi', fill='=').iter(range(n_files)):
        midi_file = midi_file_list[i]
        try:
            print(' ', end='[{}]'.format(midi_file), flush=True)
            fm = fastmidi.FastMidi(midi_file)
        except Exception as e:
            print('  Exception occurred: ' + str(e))
            continue

        tokens = fm.encode_to_tokens(sample_freq=sample_freq)
        if len(tokens) < 128:
            print('  Skipping short midi file...')
            continue
        fm.decode_from_tokens(tokens, sample_freq=sample_freq)
        outfn, _ = splitext(basename(midi_file))
        fm.write(join(midi_outdir, '{}_{}.mid'.format(cnt, outfn)))
        with open(join(midi_outdir, '{}_{}.json'.format(cnt, outfn)), 'w') as f:
            json.dump(tokens, f)
        cnt += 1


class MusicLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def load_midi(self, midi_file, midi_class=None):
        try:
            fm = fastmidi.FastMidi(midi_file)
        except Exception as e:
            print('  Exception occurred: ' + str(e))
            return None, None

        tokens = fm.encode_to_tokens(sample_freq=self.config.sample_freq)

        if len(tokens) < 64:
            print('  Skipping short midi file...')
            return None, None
        if midi_class is not None:
            tokens = [midi_class] + tokens
            # if midi_class in fastmidi.CLASS_TOKENS_CLASSICAL:
            #     tokens = [fastmidi.TOKEN_CLS_CLASSICAL, midi_class] + tokens
            # else:
            #     tokens = [midi_class] + tokens
        token_timesteps = fastmidi.encode_token_timesteps(tokens)
        return tokens, token_timesteps

    def load_prompt(self, midi_file_or_tokens, n_beats=None, style=None):
        '''Prepares a seed sequence from a midi file
        Args:
            midi_file_or_tokens: prompt midi file or token list
        Returns:
            sequence: sequence (inseq_len, n_notes)
        '''
        if type(midi_file_or_tokens) is list:
            #if any(tok not in self.tokenizer.encoder for tok in midi_file_or_tokens):
            #    sys.exit('Prompt token list contains unknown tokens')

            token_ids = self.tokenizer.convert_tokens_to_ids(midi_file_or_tokens)
            token_timesteps = fastmidi.encode_token_timesteps(midi_file_or_tokens)
        else:
            if midi_file_or_tokens.endswith('.json'):
                with open(midi_file_or_tokens, 'r') as f:
                    tokens = json.load(f)
                    token_timesteps = fastmidi.encode_token_timesteps(tokens)
            else:
                tokens, token_timesteps = self.load_midi(midi_file_or_tokens)
                if style is not None:
                    tokens = json.loads(style) + tokens

            if n_beats is None:
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                idx = np.where(token_timesteps>=(n_beats*self.config.sample_freq))
                n = idx[0][0]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens[:n])
                token_timesteps = token_timesteps[:n]
        return token_ids, token_timesteps


class MusicDataset(Dataset):
    def __init__(self, config, tokenizer, feature_dir, block_size=1200, midi_dir=None):
        '''
        '''
        self.examples = []
        if midi_dir is not None:
            loader = MusicLoader(config, tokenizer)
            all_token_ids, all_token_timesteps = [], []
            if not exists(feature_dir):
                os.makedirs(feature_dir)
            metafile = join(midi_dir, 'dataset.json')
            if exists(metafile):
                with open(metafile) as json_file:
                    metadata = json.load(json_file)
            else:
                metadata = {'.': None}
            
            for key in metadata:
                if key == '.' or key == './':
                    mdir = midi_dir
                else:
                    mdir = join(midi_dir, key)
                midi_file_list = list_all_files(mdir, ('.mid','.midi','.MID','.MIDI'))
                n_files = len(midi_file_list)
                if n_files == 0:
                    continue
                midi_class = metadata[key]
                for i in Bar('Processing Midi', fill='=').iter(range(n_files)):
                    midi_file = midi_file_list[i]
                    print(' ', end='[{}]'.format(midi_file), flush=True)
                    tokens, token_timesteps = loader.load_midi(midi_file, midi_class=midi_class)
                    if tokens is None:
                        continue

                    # Load note tokens from each file
                    for k in config.key_shifts:
                        token_ids = tokenizer.convert_tokens_to_ids(fastmidi.transpose_notes(tokens, k))
                        all_token_ids.extend(token_ids)
                        all_token_timesteps.extend(token_timesteps)
            step_size = block_size
            for i in range(0, len(all_token_ids)-block_size+1, step_size):  # Truncate in block of block_size
                #self.examples.append(all_token_ids[i:i+block_size])
                example = np.stack((all_token_ids[i:i+block_size], all_token_timesteps[i:i+block_size]), axis=1)
                self.examples.append(example)

            with open(join(feature_dir, get_dataset_name(config)+'.pkl'), "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        else:
            feature_file_list = list_all_files(feature_dir, ('.pkl'))
            for dataset in feature_file_list:
                print('[{}]'.format(dataset))
                with open(dataset, "rb") as f:
                    self.examples.extend(pickle.load(f))
                    print('  Total {} samples'.format(len(self.examples)))


    def __repr__(self):
        return '<class MusicDataset has {} examples>'.format(len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])
        #return self.examples[item] # torch.tensor(self.examples[item])

    