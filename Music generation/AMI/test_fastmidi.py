#!/usr/bin/env python3

import fastmidi
import matplotlib
import matplotlib.pyplot as plt
import sys
import json


midi_file = 'data/midi/examples/oxygene4.mid'

sample_freq = 12

print('-'*80)
print('Loaded MIDI')
fm = fastmidi.FastMidi(midi_file)

'''
fm.print() 
plt.figure(figsize=(16, 8))
plt.subplot(211)
fm.plot()
'''

print('-'*80)
print('Encoded MIDI to tokens')
tokens = fm.encode_to_tokens(sample_freq=sample_freq)

# Transpose by 2 keys, eg C Major to E Major
tokens = fastmidi.transpose_notes(tokens, 4)

print(tokens)

with open(midi_file.replace('.mid', '.json'), 'w') as f:
    json.dump(tokens, f)

print('-'*80)
print('Decoded from tokens to MIDI')
fm.decode_from_tokens(tokens, sample_freq=sample_freq)
fm.print()
fm.write(midi_file.replace('.mid', '.decoded.mid'))

fastmidi.plot(midi_file, fig_size=(16,16), fig_file=midi_file.replace('.mid', '.png'))
