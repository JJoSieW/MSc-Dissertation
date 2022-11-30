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
''' Utility functions for handling MIDI files. Adapted from PrettyMidi
'''

from __future__ import print_function

import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six

from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
                         PitchBend, ControlChange, Tempo)
from .utilities import (key_number_to_key_name, key_name_to_key_number, qpm_to_bpm, tempo_to_bpm, note_number_to_name, nearest_multiple)
from .constants import INSTRUMENT_MAP

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pypianoroll

# The largest we'd ever expect a tick to be
MAX_TICK = 1e7
SAMPLE_FREQ = 12 # timesteps per beat (quarter note)
NOTE_RANGE = (21, 100) # A0-C8
MAX_TIME_BEATS = 8 # wait time in beats
N_VELOCITIES = 8 # quantise velocity to 8 bins
TEMPO_MIN = 40
TEMPO_MAX = 240
TEMPO_STEP = 4
N_KEYS = 24
N_INSTRUMENTS = 129 # 128 is DRUMS

# Special tokens
TOKEN_BOS = '<BOS>'
TOKEN_EOS = '<EOS>'
SPECIAL_TOKENS = [
    TOKEN_BOS, 
    TOKEN_EOS, 
]
# MIDI Event tokens
TOKEN_VELOCITY = 'vel'
VELOCITY_TOKENS = ['{}:{}'.format(TOKEN_VELOCITY, v) for v in range(128//N_VELOCITIES//2, 128, 128//N_VELOCITIES)]
TOKEN_TEMPO = 'tempo'
TEMPO_TOKENS = ['{}:{}'.format(TOKEN_TEMPO, bpm) for bpm in range(TEMPO_MIN, TEMPO_MAX+1, TEMPO_STEP)]
TOKEN_KEYSIG = 'keysig'
KEYSIG_TOKENS = ['{}:{}'.format(TOKEN_KEYSIG, k) for k in range(N_KEYS)]
TOKEN_WAIT = 'wait'
TOKEN_NOTE = 'note'
# Time signatures in MIDI are often incorrect. Decided not to encode time signatures
#TOKEN_TIMESIG = 'timesig'
#TIME_SIGNATURES = ['2:2', '1:4', '2:4', '3:4', '4:4', '5:4', '6:4', '1:8', '3:8', '6:8', '7:8', '9:8', '12:8']
TOKEN_INSTRUMENT = 'instr'
INSTRUMENT_TOKENS = ['{}:{}'.format(TOKEN_INSTRUMENT, i) for i in range(N_INSTRUMENTS)]
TOKEN_PIANO = '{}:0'.format(TOKEN_INSTRUMENT)
TOKEN_DRUM = '{}:128'.format(TOKEN_INSTRUMENT)

# Class tokens
TOKEN_CLS_Q1 = '<Q1>'
TOKEN_CLS_Q2 = '<Q2>'
TOKEN_CLS_Q3 = '<Q3>'
TOKEN_CLS_Q4 = '<Q4>'
EMOTION_TOKENS = [
    TOKEN_CLS_Q1,
    TOKEN_CLS_Q2,
    TOKEN_CLS_Q3,
    TOKEN_CLS_Q4,
]

# TOKEN_CLS_BACH = '<Bach>'
# TOKEN_CLS_BEETHOVEN = '<Beethoven>'
# TOKEN_CLS_BRAHMS = '<Brahms>'
# TOKEN_CLS_CHOPIN = '<Chopin>'
# TOKEN_CLS_DEBUSSY = '<Debussy>'
# TOKEN_CLS_HANDEL = '<Handel>'
# TOKEN_CLS_HAYDN = '<Haydn>'
# TOKEN_CLS_LISZT = '<Liszt>'
# TOKEN_CLS_MENDELSSOHN = '<Mendelssohn>'
# TOKEN_CLS_MOZART = '<Mozart>'
# TOKEN_CLS_RACHMANINOFF = '<Rachmaninoff>'
# TOKEN_CLS_SCARLATTI = '<Scarlatti>'
# TOKEN_CLS_SCHUBERT = '<Schubert>'
# TOKEN_CLS_SCHUMANN = '<Schumann>'
# TOKEN_CLS_TCHAIKOVSKY = '<Tchaikovsky>'
# TOKEN_CLS_VIVALDI = '<Vivaldi>'
# CLASS_TOKENS_CLASSICAL = [
#     TOKEN_CLS_BACH,
#     TOKEN_CLS_BEETHOVEN,
#     TOKEN_CLS_BRAHMS,
#     TOKEN_CLS_CHOPIN,
#     TOKEN_CLS_DEBUSSY,
#     TOKEN_CLS_HANDEL,
#     TOKEN_CLS_HAYDN,
#     TOKEN_CLS_LISZT,
#     TOKEN_CLS_MENDELSSOHN,
#     TOKEN_CLS_MOZART,
#     TOKEN_CLS_RACHMANINOFF,
#     TOKEN_CLS_SCARLATTI,
#     TOKEN_CLS_SCHUBERT,
#     TOKEN_CLS_SCHUMANN,
#     TOKEN_CLS_TCHAIKOVSKY,
#     TOKEN_CLS_VIVALDI,
# ]
TOKEN_CLS_CLASSICAL = '<Classical>'
TOKEN_CLS_ROCK = '<Rock>'
TOKEN_CLS_ELECTRONIC = '<Electronic>'
TOKEN_CLS_JAZZ = '<Jazz>'
GENRE_TOKENS = [
    TOKEN_CLS_CLASSICAL,
    TOKEN_CLS_ROCK,
    TOKEN_CLS_ELECTRONIC,
    TOKEN_CLS_JAZZ,
]
# CLASS_TOKENS = CLASS_TOKENS_CLASSICAL + GENRE_TOKENS

CLASS_TOKENS = GENRE_TOKENS + EMOTION_TOKENS

# Default instruments to be used for each class, if not specified
GENRE_INSTRUMENTS = {
    TOKEN_CLS_CLASSICAL: [0, 1, 2, 6, 7, 8, 9, 11, 13, 19, 24, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 56, 57, 58, 59, 60, 61, 68, 69, 70, 71, 72, 73],
    TOKEN_CLS_ROCK: [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 16, 17, 18, 19, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89, 90, 91, 94, 95, 99, 100, 102, 103, 104, 119, 120, 122, 126, 128],
    TOKEN_CLS_ELECTRONIC: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 106, 108, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128],
    TOKEN_CLS_JAZZ: [0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 16, 17, 18, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 45, 48, 49, 50, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 71, 72, 73, 75, 78, 81, 82, 87, 89, 105, 119, 128],
}
def instrument_to_token(instr):
    '''Convert an instrument object to an instrument token
    '''
    i = 128 if instr.is_drum else instr.program
    return '{}:{}'.format(TOKEN_INSTRUMENT, i)

'''
    if instr.program <= 23:
        return TOKEN_PIANO # piano-like
    if instr.program >= 24 and instr.program <= 31:
        return TOKEN_GUITAR # guitar-like
    if instr.program >= 32 and instr.program <= 39:
        return TOKEN_BASS # bass
    if instr.program >= 40 and instr.program <= 55:
        return TOKEN_STRINGS # strings
    if instr.program >= 56 and instr.program <= 63:
        return TOKEN_BRASS # brass instrument
    if instr.program >= 64 and instr.program <= 71:
        return TOKEN_REED # reed instruments
    if instr.program >= 72 and instr.program <= 79:
        return TOKEN_PIPE # flute-like
    return TOKEN_REED
'''

def token_to_instrument(token):
    '''Convert an instrument token to an instrument object
    '''
    val = token.split(':')
    i = int(val[1])
    if i == 128:
        return Instrument(program=9, is_drum=True, name='Drum')
    else:
        return Instrument(program=i, is_drum=False, name=INSTRUMENT_MAP[i])

def quantise_velocity(vel):
    vstep = 128 // N_VELOCITIES
    half_step = vstep // 2
    return vel // vstep * vstep + half_step

def quantise_tempo(bpm):
    if bpm < TEMPO_MIN:
        return TEMPO_MIN
    if bpm > TEMPO_MAX:
        return TEMPO_MAX
    return bpm // TEMPO_STEP * TEMPO_STEP

def is_keysig_token(token):
    if token.startswith(TOKEN_KEYSIG + ':'):
        return True
    else:
        return False

'''
def is_timesig_token(token):
    if token.startswith(TOKEN_TIMESIG + ':'):
        return True
    else:
        return False
'''

def is_tempo_token(token):
    if token.startswith(TOKEN_TEMPO + ':'):
        return True
    else:
        return False

def is_velocity_token(token):
    if token.startswith(TOKEN_VELOCITY + ':'):
        return True
    else:
        return False

def is_note_token(token):
    if token.startswith(TOKEN_NOTE + ':'):
        return True
    else:
        return False

def is_wait_token(token):
    if token.startswith(TOKEN_WAIT + ':'):
        return True
    else:
        return False

def is_instrument_token(token):
    if token.startswith(TOKEN_INSTRUMENT + ':'):
        return True
    else:
        return False

def is_drum_token(token):
    return token == TOKEN_DRUM

def is_special_token(token):
    return token in SPECIAL_TOKENS

def is_class_token(token):
    return token in CLASS_TOKENS
    
def transpose_notes(tokens, keyshift):
    '''Transpose a list of tokens with a key shift
    '''
    if keyshift == 0:
        return tokens
    new_tokens = list(tokens)
    cur_instr = ''
    for i,tk in enumerate(new_tokens):
        # Skip instrument tokens
        if is_instrument_token(tk):
            cur_instr = tk
            continue
        # Skip drum note tokens
        if is_drum_token(cur_instr):
            continue

        val = tk.split(':')
        if is_keysig_token(tk):
            k = int(val[1]) # [0, 11] Major, [12, 23] minor
            key_idx = (k+keyshift) % 12
            mode = k // 12
            new_tokens[i] = TOKEN_KEYSIG + ':' + str(key_idx+mode*12)
        elif is_note_token(tk):
            n = int(val[1]) + keyshift
            while n < NOTE_RANGE[0]:
                n += 12
            while n > NOTE_RANGE[1]:
                n -= 12
            new_tokens[i] = TOKEN_NOTE + ':' + str(n) + ':' + val[2]
    return new_tokens

def keysig_token_to_key_name(token):
    if is_keysig_token(token):
        val = token.split(':')
        return key_number_to_key_name(int(val[1])).replace(' ', '')
    else:
        return None

def count_token_timestep(token):
    val = token.split(':')
    if val[0] == TOKEN_WAIT:
        return int(val[1])
    else:
        return 0

def encode_token_timesteps(tokens):
    '''Encode time step positions
        Args:
            tokens: list 
                a list of tokens
        Returns:
            token_timesteps: numpy array
                time step starting from 0
    '''
    token_timesteps = np.zeros(len(tokens), dtype=np.int)
    cum_steps = 0
    for i,tok in enumerate(tokens):
        token_timesteps[i] = cum_steps
        vals = tok.split(':')
        if vals[0] == TOKEN_WAIT:
            cum_steps += int(vals[1])
    return token_timesteps

def plot(midi_file, fig_size=None, fig_file=None, beats_per_bar=4):
    mt = pypianoroll.Multitrack(midi_file)
    if fig_size is None:
        nt = len(mt.tracks)
        if nt > 6:
            fig_size = (24,2*nt)
        else:
            fig_size = (24,3*nt)
    mt.beat_resolution *= beats_per_bar
    plt.rcParams.update({'font.size': 12})
    fig,ax = mt.plot(grid='x')
    plt.setp(ax, ylim=[25,107])
    plt.xlabel('Bar ({} beats)'.format(beats_per_bar))
    fig.set_size_inches(fig_size)
    if fig_file is not None:
        fig.savefig(fig_file, bbox_inches='tight')
    return fig


class FastMidi(object):
    '''A container for MIDI data in an easily-manipuaalable format.
    
    Parameters
    ----------
    midi_file : str or file
        Path or file pointer to a MIDI file.
        Default ``None`` which means create an empty class with the supplied
        values for resolution and initial tempo.
    ticks_per_beat : int
        ticks_per_beat of the MIDI data, when no file is provided.
    initial_tempo : float
        Initial tempo for the MIDI data, when no file is provided.

    Attributes
    ----------
    instruments : list
        List of :class:`fastmidi.Instrument` objects.
    tempo_changes : list
        List of :class:`fastmidi.Tempo` objects.
    key_signature_changes : list
        List of :class:`fastmidi.KeySignature` objects.
    time_signature_changes : list
        List of :class:`fastmidi.TimeSignature` objects.
    lyrics : list
        List of :class:`fastmidi.Lyric` objects.
    '''

    def __init__(self, midi_file=None, quantising=False, ticks_per_beat=480):
        '''Initialize either by populating it with MIDI data from a file or
        from scratch with no data.

        '''
        if midi_file is not None:
            # Load in the MIDI data using the midi module
            if isinstance(midi_file, six.string_types):
                # If a string was given, pass it as the string filename
                midi_data = mido.MidiFile(filename=midi_file, clip=True)
            else:
                # Otherwise, try passing it in as a file pointer
                midi_data = mido.MidiFile(file=midi_file, clip=True)

            # Convert tick values in midi_data to absolute, a useful thing.
            for track in midi_data.tracks:
                tick = 0
                for event in track:
                    event.time += tick
                    tick = event.time

            # Store the ticks_per_beat for later use
            self.ticks_per_beat = midi_data.ticks_per_beat

            # Update the array which maps ticks to time
            max_tick = max([max([e.time for e in t])
                            for t in midi_data.tracks]) + 1
            # If max_tick is huge, the MIDI file is probably corrupt
            # and creating the __tick_to_time array will thrash memory
            if max_tick > MAX_TICK:
                raise ValueError(('MIDI file has a largest tick of {},'
                                    ' it is likely corrupt'.format(max_tick)))

            # Populate the list of key and time signature changes
            self._load_metadata(midi_data)

            # Check that there are tempo, key and time change events
            # only on track 0
            if any(e.type in ('set_tempo', 'key_signature', 'time_signature')
                    for track in midi_data.tracks[1:] for e in track):
                warnings.warn(
                    'Tempo, Key or Time signature change events found on '
                    'non-zero tracks.  This is not a valid type 0 or type 1 '
                    'MIDI file.  Tempo, Key or Time Signature may be wrong.',
                    RuntimeWarning)

            # Populate the list of instruments
            self._load_instruments(midi_data)

            # Snaping to beats
            if quantising:
                self._quantise()
        else:
            self.clear()
            self.ticks_per_beat = ticks_per_beat

    def _load_metadata(self, midi_data):
        '''Populates ``self.time_signature_changes`` with ``TimeSignature``
        objects, ``self.key_signature_changes`` with ``KeySignature`` objects,
        and ``self.lyrics`` with ``Lyric`` objects.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        '''

        # Initialize empty lists for storing key signature changes, time
        # signature changes, and lyrics
        self.key_signature_changes = []
        self.time_signature_changes = []
        self.lyrics = []
        self.tempo_changes = []

        for event in midi_data.tracks[0]:
            if event.type == 'key_signature':
                key_obj = KeySignature(
                    key_name_to_key_number(event.key),
                    self._ticks_to_beats(event.time))
                self.key_signature_changes.append(key_obj)

            elif event.type == 'time_signature':
                ts_obj = TimeSignature(event.numerator,
                                       event.denominator,
                                       self._ticks_to_beats(event.time))
                self.time_signature_changes.append(ts_obj)

            elif event.type == 'lyrics':
                self.lyrics.append(Lyric(
                    event.text, self._ticks_to_beats(event.time)))

            elif event.type == 'set_tempo':
                self.tempo_changes.append(Tempo(tempo_to_bpm(event.tempo), self._ticks_to_beats(event.time)))
                    
    def _load_instruments(self, midi_data):
        '''Populates ``self.instruments`` using ``midi_data``.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        '''
        # MIDI files can contain a collection of tracks; each track can have
        # events occuring on one of sixteen channels, and events can correspond
        # to different instruments according to the most recently occurring
        # program number.  So, we need a way to keep track of which instrument
        # is playing on each track on each channel.  This dict will map from
        # program number, drum/not drum, channel, and track index to instrument
        # indices, which we will retrieve/populate using the __get_instrument
        # function below.
        instrument_map = collections.OrderedDict()
        # Store a similar mapping to instruments storing 'straggler events',
        # e.g. events which appear before we want to initialize an Instrument
        stragglers = {}
        # This dict will map track indices to any track names encountered
        track_name_map = collections.defaultdict(str)

        def __get_instrument(program, channel, track, create_new):
            '''Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            '''
            # If we have already created an instrument for this program
            # number/track/channel, return it
            if (program, channel, track) in instrument_map:
                return instrument_map[(program, channel, track)]
            # If there's a straggler instrument for this instrument and we
            # aren't being requested to create a new instrument
            if not create_new and (channel, track) in stragglers:
                return stragglers[(channel, track)]
            # If we are told to, create a new instrument and store it
            if create_new:
                is_drum = (channel == 9)
                instrument = Instrument(
                    program, is_drum, track_name_map[track_idx])
                # If any events appeared for this instrument before now,
                # include them in the new instrument
                if (channel, track) in stragglers:
                    straggler = stragglers[(channel, track)]
                    instrument.control_changes = straggler.control_changes
                    instrument.pitch_bends = straggler.pitch_bends
                # Add the instrument to the instrument map
                instrument_map[(program, channel, track)] = instrument
            # Otherwise, create a 'straggler' instrument which holds events
            # which appear before we actually want to create a proper new
            # instrument
            else:
                # Create a 'straggler' instrument
                instrument = Instrument(program, track_name_map[track_idx])
                # Note that stragglers ignores program number, because we want
                # to store all events on a track which appear before the first
                # note-on, regardless of program
                stragglers[(channel, track)] = instrument
            return instrument

        for track_idx, track in enumerate(midi_data.tracks):
            # Keep track of last note on location:
            # key = (instrument, note),
            # value = (note-on tick, velocity)
            last_note_on = collections.defaultdict(list)
            # Keep track of which instrument is playing in each channel
            # initialize to program 0 for all channels
            current_instrument = np.zeros(16, dtype=np.int)
            for event in track:
                # Look for track name events
                if event.type == 'track_name':
                    # Set the track name for the current track
                    track_name_map[track_idx] = event.name
                # Look for program change events
                if event.type == 'program_change':
                    # Update the instrument for this channel
                    current_instrument[event.channel] = event.program
                # Note ons are note on events with velocity > 0
                elif event.type == 'note_on' and event.velocity > 0:
                    # Store this as the last note-on location
                    note_on_index = (event.channel, event.note)
                    last_note_on[note_on_index].append((event.time, event.velocity))
                # Note offs can also be note on events with 0 velocity
                elif event.type == 'note_off' or (event.type == 'note_on' and event.velocity == 0):
                    # Check that a note-on exists (ignore spurious note-offs)
                    key = (event.channel, event.note)
                    if key in last_note_on:
                        # Get the start/stop times and velocity of every note
                        # which was turned on with this instrument/drum/pitch.
                        # One note-off may close multiple note-on events from
                        # previous ticks. In case there's a note-off and then
                        # note-on at the same tick we keep the open note from
                        # this tick.
                        end_tick = event.time
                        open_notes = last_note_on[key]

                        notes_to_close = [
                            (start_tick, velocity)
                            for start_tick, velocity in open_notes
                            if start_tick != end_tick]
                        notes_to_keep = [
                            (start_tick, velocity)
                            for start_tick, velocity in open_notes
                            if start_tick == end_tick]

                        for start_tick, velocity in notes_to_close:
                            offset = self._ticks_to_beats(start_tick)
                            duration = self._ticks_to_beats(end_tick-start_tick)
                            if duration > 0:
                                # Create the note event
                                note = Note(event.note, offset, duration, velocity)
                                # Get the program and drum type for the current
                                # instrument
                                program = current_instrument[event.channel]
                                # Retrieve the Instrument instance for the current
                                # instrument
                                # Create a new instrument if none exists
                                instrument = __get_instrument(
                                    program, event.channel, track_idx, 1)
                                # Add the note event
                                instrument.notes.append(note)

                        if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                            # Note-on on the same tick but we already closed
                            # some previous notes -> it will continue, keep it.
                            last_note_on[key] = notes_to_keep
                        else:
                            # Remove the last note on for this instrument
                            del last_note_on[key]
                # Store pitch bends
                elif event.type == 'pitchwheel':
                    # Create pitch bend class instance
                    bend = PitchBend(event.pitch, self._ticks_to_beats(event.time))
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(
                        program, event.channel, track_idx, 0)
                    # Add the pitch bend event
                    instrument.pitch_bends.append(bend)
                # Store control changes
                elif event.type == 'control_change':
                    control_change = ControlChange(event.control, event.value, self._ticks_to_beats(event.time))
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(
                        program, event.channel, track_idx, 0)
                    # Add the control change event
                    instrument.control_changes.append(control_change)
        # Initialize list of instruments from instrument_map
        self.instruments = [i for i in instrument_map.values()]

    def _quantise(self, beat_divisors=([4,6]), process_offsets=True, process_durations=True):
        '''Quantise time values by snapping offsets and/or durations to the 
        nearest multiple of a beat value given as one or more divisors of 1 beat.
        '''
        def __best_match(target, divisors):
            found = []
            for div in divisors:
                match, error, signedError = nearest_multiple(target, (1.0/div))
                found.append((error, match, signedError))  # reverse for sorting
            # get first, and leave out the error
            bestMatchTuple = sorted(found)[0]
            return bestMatchTuple[1]

        # Quantise offsets and durations of all instruments
        for i in self.instruments:
            for n in i.notes:
                if process_offsets:
                    n.offset = __best_match(float(n.offset), beat_divisors)
                if process_durations:
                    n.duration = __best_match(float(n.duration), beat_divisors)
            for b in i.pitch_bends:
                if process_offsets:
                    b.offset = __best_match(float(b.offset), beat_divisors)
            for c in i.control_changes:
                if process_offsets:
                    c.offset = __best_match(float(c.offset), beat_divisors)
        
        # Quantise offsets of all meta-events
        if process_offsets:
            for m in self.tempo_changes:
                m.offset = __best_match(float(m.offset), beat_divisors)
            for m in self.time_signature_changes:
                m.offset = __best_match(float(m.offset), beat_divisors)
            for m in self.key_signature_changes:
                m.offset = __best_match(float(m.offset), beat_divisors)
            for m in self.lyrics: 
                m.offset = __best_match(float(m.offset), beat_divisors)

    def get_duration(self):
        '''Returns the total duration of this MIDI.

        Returns
        -------
        duration : float
            Duration, in beat, of this instrument.

        '''
        durations = [i.get_duration() for i in self.instruments]
        # If there are no events, return 0
        if len(durations) == 0:
            return 0.
        else:
            return max(durations)

    def encode_to_tokens(self, sample_freq=SAMPLE_FREQ):
        '''Encode midi into tokens
        Args:
            sample_freq: int
                The number of samples per quarter note (default 12)
        Returns:
            tokens: list of strings
                A list of tokens representing instrument, note and rest events
                'wait:'      wait event token, followed by duration in timesteps (beat * sample_freq)
                'note:'      note event token, e.g. note_60_12 represents piano note 60, lasting 12 time steps
                'vel:'       velocity event token, followed by velocity between [1, 127]
                'tempo:'     tempo change event token, followed by BMP
                'keysig:'    key signature event, followed by number [0, 11] Major, [12, 23] minor
                                ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
                'timesig:'   time signature event, followed by numerator and denominator, e.g. 3/4 is 'timesig:3:4'
                e.g. ['<BOS>', 'keysig:0', 'tempo:80', '<Piano>', 'vel:88', 'note:60:12', 'note:62:12', 'wait:12', 'note:60:12', '<EOS>']
        '''
        # Generate velocity bins
        max_time_step = round(self.get_duration() * sample_freq)
        
        # timestep to event dict
        ts2events = {}
        
        # Process key signaure events
        for keysig in self.key_signature_changes:
            ts = round(keysig.offset*sample_freq)
            if ts > max_time_step:
                continue
            e = '{}_{}'.format(TOKEN_KEYSIG, keysig.key_number)
            if ts in ts2events:
                ts2events[ts].append(e)
            else:
                ts2events[ts] = [e]

        # Process time signaure events
        # Time signatures in MIDI are often incorrect. Decided not to encode time signatures
        '''
        for timesig in self.time_signature_changes:
            ts = round(timesig.offset*sample_freq)
            if ts > max_time_step:
                continue
            e = '{}:{}:{}'.format(TOKEN_TIMESIG, timesig.numerator, timesig.denominator)
            if ts in ts2events:
                ts2events[ts].append(e)
            else:
                ts2events[ts] = [e]
        '''

        # Process tempo events
        for tempo in self.tempo_changes:
            ts = round(tempo.offset*sample_freq)
            if ts > max_time_step:
                continue
            e = '{}_{}'.format(TOKEN_TEMPO, quantise_tempo(tempo.bpm))
            if ts in ts2events:
                ts2events[ts].append(e)
            else:
                ts2events[ts] = [e]

        # Process all note events and sort them according to time steps
        for instr in self.instruments:
            # Skip instrument if not listed in instruments
            ins_tok = instrument_to_token(instr)
            if not is_instrument_token(ins_tok):
                continue
            for note in instr.notes:
                p = note.pitch
                while p < NOTE_RANGE[0]:
                    p += 12
                while p > NOTE_RANGE[1]:
                    p -= 12
                # note events
                ts = round(note.offset*sample_freq)
                dur = round(note.duration*sample_freq)
                e = '{}_{}_{}_{}'.format(ins_tok, p, dur, quantise_velocity(note.velocity))
                if ts in ts2events:
                    ts2events[ts].append(e)
                else:
                    ts2events[ts] = [e]
            
        # Encode events to a list of tokens
        max_wait_time = MAX_TIME_BEATS * sample_freq
        all_tokens = [TOKEN_BOS]
        prev_ts = 0
        prev_vel = ''
        prev_instr = ''
        prev_tempo = ''
        for ts in sorted(ts2events.keys()):
            # Add wait tokens
            if ts > prev_ts:
                wait_count = ts - prev_ts
                while wait_count > max_wait_time:
                    all_tokens.append('{}:{}'.format(TOKEN_WAIT, max_wait_time))
                    wait_count -= max_wait_time
                if wait_count > 0:
                    all_tokens.append('{}:{}'.format(TOKEN_WAIT, wait_count))
                prev_ts = ts

            # Add key signature and tempo tokens first then note tokens
            note_tokens = []
            # Add note tokens sorted by pitch
            for e in sorted(ts2events[ts]):
                val = e.split('_')
                if val[0] == TOKEN_KEYSIG: #or val[0] == TOKEN_TIMESIG:
                    all_tokens.append('{}:{}'.format(TOKEN_KEYSIG, val[1]))
                elif val[0] == TOKEN_TEMPO:
                    if val[1] != prev_tempo:
                        prev_tempo = val[1]
                        all_tokens.append('{}:{}'.format(TOKEN_TEMPO, val[1]))
                else:
                    instr = val[0]
                    if val[0] != prev_instr:
                        prev_instr = val[0]
                        note_tokens.append(val[0])
                    if val[3] != prev_vel:
                        prev_vel = val[3]
                        note_tokens.append('{}:{}'.format(TOKEN_VELOCITY, val[3]))
                    note_tokens.append('{}:{}:{}'.format(TOKEN_NOTE, val[1], val[2]))

            all_tokens.extend(note_tokens)

        all_tokens.append(TOKEN_EOS)
        return all_tokens

    def decode_from_tokens(self, tokens, sample_freq=SAMPLE_FREQ, bpm=80):
        '''Decode tokens into MIDI
        Args:
            tokens: list of strings
                A list of tokens representing instrument, note and rest events
                'wait:'      wait event token, followed by duration in timesteps (beat * sample_freq)
                'note:'      note event token, e.g. note_60_12 represents piano note 60, lasting 12 time steps
                'vel:'       velocity event token, followed by velocity between [1, 127]
                'tempo:'     tempo change event token, followed by BMP
                'keysig:'    key signature event, followed by number [0, 11] Major, [12, 23] minor
                                ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
                'timesig:'   time signature event, followed by numerator and denominator, e.g. 3/4 is 'timesig:3:4'
                e.g. ['<BOS>', 'key:0', 'tempo:80', '<Piano>', 'vel:88', 'note:60:12', 'note:62:12', 'wait:12', 'note:60:12', '<EOS>']
            sample_freq: int
                The number of samples per quarter note (default 12)
            bpm: int
                default BPM if no tempo event found
        Returns:
            
        '''
        self.clear()
        instruments = {}

        # Parse tokens to notes
        cur_beat = 0
        cur_vel = 80
        cur_instr = TOKEN_PIANO
        open_notes = {}
        for tk in tokens:
            if is_special_token(tk):
                continue
            if is_class_token(tk):
                continue
            if is_instrument_token(tk):
                cur_instr = tk
            val = tk.split(':')
            if val[0] == TOKEN_VELOCITY:
                cur_vel = int(val[1])
            elif val[0] == TOKEN_KEYSIG:
                self.key_signature_changes.append(KeySignature(int(val[1]), cur_beat))
            #elif val[0] == TOKEN_TIMESIG:
            #    self.time_signature_changes.append(TimeSignature(int(val[1]), int(val[2]), cur_beat))
            elif val[0] == TOKEN_TEMPO:
                self.tempo_changes.append(Tempo(int(val[1]), cur_beat))
            elif val[0] == TOKEN_NOTE:
                instr_note = val[1] + ':' + val[2]
                open_notes[instr_note] = [cur_beat, cur_vel]
                # Create Instrument object
                if cur_instr not in instruments:
                    instruments[cur_instr] = token_to_instrument(cur_instr)
                pitch = int(val[1])
                duration = int(val[2]) / sample_freq
                instruments[cur_instr].notes.append(Note(pitch, cur_beat, duration, cur_vel))
            elif val[0] == TOKEN_WAIT:
                cur_beat += int(val[1]) / sample_freq
                
        # Add instruments that are not empty
        for k in instruments:
            if len(instruments[k].notes) > 0:
                self.instruments.append(instruments[k])

        # Set BPM
        if len(self.tempo_changes) == 0:
            self.tempo_changes.append(Tempo(bpm, 0))

    def clear(self):
        # Empty instruments
        self.instruments = []
        # Empty tempo changes list
        self.tempo_changes = []
        # Empty key signature changes list
        self.key_signature_changes = []
        # Empty time signatures changes list
        self.time_signature_changes = []
        # Empty lyrics list
        self.lyrics = []

    def get_offsets(self):
        '''Return a sorted list of the times of all offsets of all notes from
        all instruments.  May have duplicate entries.

        Returns
        -------
        offsets : np.ndarray
            Offsets of all notes in beats.

        '''
        offsets = np.array([])
        # Just concatenate onsets from all the instruments
        for instrument in self.instruments:
            offsets = np.append(offsets, instrument.get_offsets())
        # Return them sorted (because why not?)
        return np.sort(offsets)

    def print(self):
        for i, ins in enumerate(self.instruments):
            print('Program {} ({})'.format(i, ins.name))
            ins.print()

    def plot(self, sample_freq=SAMPLE_FREQ, plot_file=None, beats=None):
        '''Plot a combined piano roll
        '''
        piano_roll = self.get_piano_roll(sample_freq=sample_freq)
        if beats is not None:
            if isinstance(beats, int) or len(beats) == 1:
                piano_roll = piano_roll[:,:beats*sample_freq+1]
            elif len(beats) == 2:
                t1 = beats[0] * sample_freq
                t2 = beats[1] * sample_freq + 1
                piano_roll = piano_roll[:,t1:t2]

        piano_roll = piano_roll[NOTE_RANGE[0]:NOTE_RANGE[1],:]
        #plt.figure(figsize=(20, 5))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='Blues')
        n_octaves = (NOTE_RANGE[1] - NOTE_RANGE[0]) // 12

        yticks = [0] + [o*12+n for o in range(n_octaves) for n in [4,7,12]]
        yticklabels = [note_number_to_name(n+NOTE_RANGE[0]) for n in yticks]
        xtick_step = piano_roll.shape[1] // 8
        xticks = range(0, piano_roll.shape[1], xtick_step)
        xticklabels = ['{:.1f}'.format(t/sample_freq) for t in xticks]
        plt.yticks(yticks, yticklabels)
        plt.ylim([-1, NOTE_RANGE[1] - NOTE_RANGE[0]])
        plt.xticks(xticks, xticklabels)
        plt.xlim([-sample_freq//4, piano_roll.shape[1]+sample_freq//4])
        plt.tight_layout()
        
        # Save figure
        if plot_file is not None:
            plt.savefig(plot_file, bbox_inches='tight')

    def get_piano_roll(self, sample_freq=SAMPLE_FREQ, pedal_threshold=64):
        '''Compute a piano roll matrix of the MIDI data.

        Parameters
        ----------
        sample_freq : int
            Beat sampling frequency of the timesteps, i.e. each timestep is spaced apart
            by ``1./sample_freq`` beat.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_roll : np.ndarray, shape=(128,timesteps.shape[0])
            Piano roll of MIDI data, flattened across instruments.

        '''
        # If there are no instruments, return an empty array
        if len(self.instruments) == 0:
            return np.zeros((128, 0))

        # Get piano rolls for each instrument
        piano_rolls = [i.get_piano_roll(sample_freq=sample_freq, pedal_threshold=pedal_threshold)
                       for i in self.instruments]
        # Allocate piano roll,
        # number of columns is max of # of columns in all piano rolls
        piano_roll = np.zeros((128, np.max([p.shape[1] for p in piano_rolls])))
        # Sum each piano roll into the aggregate piano roll
        for roll in piano_rolls:
            piano_roll[:, :roll.shape[1]] += roll
        return piano_roll

    def get_all_piano_rolls(self, sample_freq=SAMPLE_FREQ, pedal_threshold=64):
        '''Compute a piano roll matrix for each program in the MIDI file.

        Parameters
        ----------
        sample_freq : int
            Beat sampling frequency of the timesteps, i.e. each timestep is spaced apart
            by ``1./sample_freq`` beat.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_rolls : a list of np.ndarray, shape=(128,timesteps.shape[0])
            Piano rolls of MIDI data

        '''

        # If there are no instruments, return an empty array
        if len(self.instruments) == 0:
            return []

        # Get piano rolls for each instrument
        piano_rolls = [i.get_piano_roll(sample_freq=sample_freq, pedal_threshold=pedal_threshold)
                       for i in self.instruments]
        
        return piano_rolls

    def get_pitch_class_histogram(self, use_duration=False,
                                  use_velocity=False, normalize=True):
        '''Computes the histogram of pitch classes.

        Parameters
        ----------
        use_duration : bool
            Weight frequency by note duration.
        use_velocity : bool
            Weight frequency by note velocity.
        normalize : bool
            Normalizes the histogram such that the sum of bin values is 1.

        Returns
        -------
        histogram : np.ndarray, shape=(12,)
            Histogram of pitch classes given all tracks, optionally weighted
            by their durations or velocities.
        '''
        # Sum up all histograms from all instruments defaulting to np.zeros(12)
        histogram = sum([
            i.get_pitch_class_histogram(use_duration, use_velocity)
            for i in self.instruments], np.zeros(12))

        # Normalize accordingly
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))

        return histogram

    def get_chroma(self, sample_freq=12, timesteps=None, pedal_threshold=64):
        '''Get the MIDI data as a sequence of chroma vectors.

        Parameters
        ----------
        sample_freq : int
            Beat sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./sample_freq`` beat.
        timesteps : np.ndarray
            Timesteps of the start of each column in the piano roll.
            Default ``None`` which is ``np.arange(0, get_end_offset(), 1./sample_freq)``.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_roll : np.ndarray, shape=(12,times.shape[0])
            Chromagram of MIDI data, flattened across instruments.

        '''
        # First, get the piano roll
        piano_roll = self.get_piano_roll(sample_freq=sample_freq, pedal_threshold=pedal_threshold)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def fluidsynth(self, fs=44100, sf2_path=None):
        '''Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize at.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``fastmidi``.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.

        '''
        # If there are no instruments, or all instruments have no notes, return
        # an empty array
        if len(self.instruments) == 0 or all(len(i.notes) == 0
                                             for i in self.instruments):
            return np.array([])
        # Get synthesized waveform for each instrument
        waveforms = [i.fluidsynth(fs=fs,
                                  sf2_path=sf2_path) for i in self.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        synthesized /= np.abs(synthesized).max()
        return synthesized

    def _ticks_to_beats(self, ticks):
        # Check that the tick isn't too big
        if ticks >= MAX_TICK:
            raise IndexError('Supplied tick is too large.')
        return ticks / self.ticks_per_beat

    def _beats_to_ticks(self, beats):
        return round(beats * self.ticks_per_beat)

    def remove_invalid_notes(self):
        '''Removes any notes whose end time is before or at their start time.

        '''
        # Simply call the child method on all instruments
        for instrument in self.instruments:
            instrument.remove_invalid_notes()

    def write(self, filename):
        '''Write the MIDI data out to a .mid file.

        Parameters
        ----------
        filename : str or file
            Path or file to write .mid file to.

        '''

        def event_compare(event1, event2):
            '''Compares two events for sorting.

            Events are sorted by tick time ascending. Events with the same tick
            time ares sorted by event type. Some events are sorted by
            additional values. For example, Note On events are sorted by pitch
            then velocity, ensuring that a Note Off (Note On with velocity 0)
            will never follow a Note On with the same pitch.

            Parameters
            ----------
            event1, event2 : mido.Message
               Two events to be compared.
            '''
            # Construct a dictionary which will map event names to numeric
            # values which produce the correct sorting.  Each dictionary value
            # is a function which accepts an event and returns a score.
            # The spacing for these scores is 256, which is larger than the
            # largest value a MIDI value can take.
            secondary_sort = {
                'set_tempo': lambda e: (1 * 256 * 256),
                'time_signature': lambda e: (2 * 256 * 256),
                'key_signature': lambda e: (3 * 256 * 256),
                'lyrics': lambda e: (4 * 256 * 256),
                'program_change': lambda e: (5 * 256 * 256),
                'pitchwheel': lambda e: ((6 * 256 * 256) + e.pitch),
                'control_change': lambda e: (
                    (7 * 256 * 256) + (e.control * 256) + e.value),
                'note_off': lambda e: ((8 * 256 * 256) + (e.note * 256)),
                'note_on': lambda e: (
                    (9 * 256 * 256) + (e.note * 256) + e.velocity),
                'end_of_track': lambda e: (10 * 256 * 256)
            }
            # If the events have the same tick, and both events have types
            # which appear in the secondary_sort dictionary, use the dictionary
            # to determine their ordering.
            if (event1.time == event2.time and
                    event1.type in secondary_sort and
                    event2.type in secondary_sort):
                return (secondary_sort[event1.type](event1) -
                        secondary_sort[event2.type](event2))
            # Otherwise, just return the difference of their ticks.
            return event1.time - event2.time

        # Initialize output MIDI object
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        # Create track 0 with timing information
        timing_track = mido.MidiTrack()
        # Add a default time signature only if there is not one at offset 0.
        add_ts = True
        if self.time_signature_changes:
            add_ts = min([ts.offset for ts in self.time_signature_changes]) > 0.0
        if add_ts:
            # Add time signature event with default values (4/4)
            timing_track.append(mido.MetaMessage(
                'time_signature', time=0, numerator=4, denominator=4))

        # Add in each tempo change event
        for t in self.tempo_changes:
            timing_track.append(mido.MetaMessage(
                'set_tempo', time=self._beats_to_ticks(t.offset), tempo=t.tempo)) # microseconds per beat
                
        # Add in each time signature
        for ts in self.time_signature_changes:
            timing_track.append(mido.MetaMessage(
                'time_signature', time=self._beats_to_ticks(ts.offset),
                numerator=ts.numerator, denominator=ts.denominator))

        # Add in each key signature
        # Mido accepts key changes in a different format than fastmidi, this
        # list maps key number to mido key name
        key_number_to_mido_key_name = [
            'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am',
            'Bbm', 'Bm']
        for ks in self.key_signature_changes:
            timing_track.append(mido.MetaMessage(
                'key_signature', time=self._beats_to_ticks(ks.offset),
                key=key_number_to_mido_key_name[ks.key_number]))

        # Add in all lyrics events
        for l in self.lyrics:
            timing_track.append(mido.MetaMessage(
                'lyrics', time=self._beats_to_ticks(l.offset), text=l.text))

        # Sort the (absolute-tick-timed) events.
        timing_track.sort(key=functools.cmp_to_key(event_compare))
        # Add in an end of track event
        timing_track.append(mido.MetaMessage(
            'end_of_track', time=timing_track[-1].time + 1))
        mid.tracks.append(timing_track)

        # Create a list of possible channels to assign - this seems to matter
        # for some synths.
        channels = list(range(16))
        # Don't assign the drum channel by mistake!
        channels.remove(9)
        for n, instrument in enumerate(self.instruments):
            # Initialize track for this instrument
            track = mido.MidiTrack()
            # Add track name event if instrument has a name
            if instrument.name:
                track.append(mido.MetaMessage(
                    'track_name', time=0, name=instrument.name))
            # If it's a drum event, we need to set channel to 9
            if instrument.is_drum:
                channel = 9
            # Otherwise, choose a channel from the possible channel list
            else:
                channel = channels[n % len(channels)]
            # Set the program number
            track.append(mido.Message(
                'program_change', time=0, program=instrument.program,
                channel=channel))
            # Add all note events
            for note in instrument.notes:
                # Construct the note-on event
                track.append(mido.Message(
                    'note_on', time=self._beats_to_ticks(note.offset),
                    channel=channel, note=note.pitch, velocity=note.velocity))
                # Also need a note-off event (note on with velocity 0)
                track.append(mido.Message(
                    'note_on', time=self._beats_to_ticks(note.offset+note.duration),
                    channel=channel, note=note.pitch, velocity=0))
            # Add all pitch bend events
            for bend in instrument.pitch_bends:
                track.append(mido.Message(
                    'pitchwheel', time=self._beats_to_ticks(bend.offset),
                    channel=channel, pitch=bend.pitch))
            # Add all control change events
            for control_change in instrument.control_changes:
                track.append(mido.Message(
                    'control_change',
                    time=self._beats_to_ticks(control_change.offset),
                    channel=channel, control=control_change.number,
                    value=control_change.value))
            # Sort all the events using the event_compare comparator.
            track = sorted(track, key=functools.cmp_to_key(event_compare))

            # If there's a note off event and a note on event with the same
            # tick and pitch, put the note off event first
            for n, (event1, event2) in enumerate(zip(track[:-1], track[1:])):
                if (event1.time == event2.time and
                        event1.type == 'note_on' and
                        event2.type == 'note_on' and
                        event1.note == event2.note and
                        event1.velocity != 0 and
                        event2.velocity == 0):
                    track[n] = event2
                    track[n + 1] = event1
            # Finally, add in an end of track event
            track.append(mido.MetaMessage('end_of_track', time=track[-1].time + 1))
            # Add to the list of output tracks
            mid.tracks.append(track)
        # Turn ticks to relative time from absolute
        for track in mid.tracks:
            tick = 0
            for event in track:
                event.time -= tick
                tick += event.time
        # Write it out
        if isinstance(filename, six.string_types):
            # If a string was given, pass it as the string filename
            mid.save(filename=filename)
        else:
            # Otherwise, try passing it in as a file pointer
            mid.save(file=filename)
