# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
""" AMI Tokenisation"""


import json
import logging
import os
from functools import lru_cache

import regex as re

from .tokenization_utils import PreTrainedTokenizer

from os.path import join, exists
import fastmidi
import numpy as np

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'ami': 2048,
}

class AmiTokenizer(PreTrainedTokenizer):
    """
    Ami Tokenizer
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        sample_freq=fastmidi.SAMPLE_FREQ, 
        bos_token=fastmidi.TOKEN_BOS,
        eos_token=fastmidi.TOKEN_EOS,
        pad_token=fastmidi.TOKEN_EOS,
        unk_token=fastmidi.TOKEN_EOS,
        errors="replace",
        **kwargs
    ):
        super(AmiTokenizer, self).__init__(
            bos_token=bos_token, 
            eos_token=eos_token, 
            unk_token=unk_token, 
            pad_token=pad_token,
            **kwargs)
        self.max_len_single_sentence = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = (
            self.max_len
        )  # no default special tokens - you can update this value if you add special tokens

        self.sample_freq = sample_freq

        # Build vocabulary
        vocab_file = kwargs.pop('vocab_file', self.vocab_files_names['vocab_file'])
        if exists(vocab_file):
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
            self.decoder = {v: k for k, v in self.encoder.items()}
        else:
            self.build_vocab()
        
        self.errors = errors  # how to handle errors in decoding

    def keysig_token_ids(self):
        '''
        t1 = '{}:{}'.format(fastmidi.TOKEN_KEYSIG, 0)
        t2 = '{}:{}'.format(fastmidi.TOKEN_KEYSIG, fastmidi.N_KEYS-1)
        ids = self.convert_tokens_to_ids([t1, t2])
        return [i for i in range(ids[0], ids[1]+1)]
        '''
        return self.convert_tokens_to_ids(fastmidi.KEYSIG_TOKENS)

    def tempo_token_ids(self):
        '''
        t1 = '{}:{}'.format(fastmidi.TOKEN_TEMPO, fastmidi.TEMPO_MIN)
        t2 = '{}:{}'.format(fastmidi.TOKEN_TEMPO, fastmidi.TEMPO_MAX)
        ids = self.convert_tokens_to_ids([t1, t2])
        return [i for i in range(ids[0], ids[1]+1)]
        '''
        return self.convert_tokens_to_ids(fastmidi.TEMPO_TOKENS)

    def velocity_token_ids(self):
        '''
        t1 = '{}:{}'.format(fastmidi.TOKEN_VELOCITY, fastmidi.quantise_velocity(1))
        t2 = '{}:{}'.format(fastmidi.TOKEN_VELOCITY, fastmidi.quantise_velocity(127))
        ids = self.convert_tokens_to_ids([t1, t2])
        return [i for i in range(ids[0], ids[1]+1)]
        '''
        return self.convert_tokens_to_ids(fastmidi.VELOCITY_TOKENS)

    def wait_token_ids(self):
        t1 = '{}:{}'.format(fastmidi.TOKEN_WAIT, 1)
        t2 = '{}:{}'.format(fastmidi.TOKEN_WAIT, fastmidi.MAX_TIME_BEATS*self.sample_freq)
        ids = self.convert_tokens_to_ids([t1, t2])
        return [i for i in range(ids[0], ids[1]+1)]

    def long_wait_token_ids(self, min_timestep):
        t1 = '{}:{}'.format(fastmidi.TOKEN_WAIT, min_timestep)
        t2 = '{}:{}'.format(fastmidi.TOKEN_WAIT, fastmidi.MAX_TIME_BEATS*self.sample_freq)
        ids = self.convert_tokens_to_ids([t1, t2])
        return [i for i in range(ids[0], ids[1]+1)]

    def note_token_ids(self):
        t1 = '{}:{}:{}'.format(fastmidi.TOKEN_NOTE, fastmidi.NOTE_RANGE[0], 1)
        t2 = '{}:{}:{}'.format(fastmidi.TOKEN_NOTE, fastmidi.NOTE_RANGE[1], fastmidi.MAX_TIME_BEATS*self.sample_freq)
        ids = self.convert_tokens_to_ids([t1, t2])
        return [i for i in range(ids[0], ids[1]+1)]
        
    def instrument_token_ids(self):
        return self.convert_tokens_to_ids(fastmidi.INSTRUMENT_TOKENS)
        
    def class_token_ids(self):
        return self.convert_tokens_to_ids(fastmidi.CLASS_TOKENS)
        
    def build_vocab(self):
        '''Build vocabulary
        '''
        # Add special tokens first
        vocab = fastmidi.SPECIAL_TOKENS.copy()
        
        # Key signature tokens
        vocab += fastmidi.KEYSIG_TOKENS

        ## Time signature tokens
        #vocab += ['{}:{}'.format(fastmidi.TOKEN_TIMESIG, n) for n in fastmidi.TIME_SIGNATURES]

        # Tempo tokens
        vocab += fastmidi.TEMPO_TOKENS
        
        # Velocity tokens
        vocab += fastmidi.VELOCITY_TOKENS
        
        # Instrument tokens
        vocab += fastmidi.INSTRUMENT_TOKENS

        # Wait tokens
        vocab += ['{}:{}'.format(fastmidi.TOKEN_WAIT, i) for i in range(1, fastmidi.MAX_TIME_BEATS*self.sample_freq+1)]
        
        # Note tokens
        vocab += ['{}:{}:{}'.format(fastmidi.TOKEN_NOTE, n, t) for n in range(fastmidi.NOTE_RANGE[0], fastmidi.NOTE_RANGE[1]+1) for t in range(1, fastmidi.MAX_TIME_BEATS*self.sample_freq+1)]
        
        # Add class tokens at the end so that we can add more classes later
        vocab += fastmidi.CLASS_TOKENS

        self.encoder = {v:i for i,v in enumerate(vocab)}
        self.decoder = {i:v for v,i in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def _tokenize(self, text):
        """ Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space to get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
        text.split(" ")
        return text.split(" ")

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = " ".join(tokens)
        return text

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        return vocab_file,

    def count_tokenid_timestep(self, token_id):
        token = self._convert_id_to_token(token_id)
        return fastmidi.count_token_timestep(token)

    """
    def count_tokenid_beat(self, token_id):
        token = self._convert_id_to_token(token_id)
        return fastmidi.count_token_timestep(token) // self.sample_freq

    def encode_token_beats(self, tokens):
        '''Encode beat positions
            Args:
                tokens: list 
                    a list of encoded tokens
            Returns:
                token_beats: numpy array
                    beat number starting from 0
        '''
        token_beats = fastmidi.encode_token_timesteps(tokens) // self.sample_freq
        return token_beats
    """
    