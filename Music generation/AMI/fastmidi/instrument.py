'''The Instrument class holds all events for a single instrument and contains
functions for extracting information from the events it contains.
'''
import numpy as np
try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False
import os
import pkg_resources

from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz

DEFAULT_SF2 = 'TimGM6mb.sf2'


class Instrument(object):
    '''Object to hold event information for a single instrument.

    Parameters
    ----------
    program : int
        MIDI program number (instrument index), in ``[0, 127]``.
    is_drum : bool
        Is the instrument a drum instrument (channel 9)?
    name : str
        Name of the instrument.

    Attributes
    ----------
    program : int
        The program number of this instrument.
    is_drum : bool
        Is the instrument a drum instrument (channel 9)?
    name : str
        Name of the instrument.
    notes : list
        List of :class:`fastmidi.Note` objects.
    pitch_bends : list
        List of of :class:`fastmidi.PitchBend` objects.
    control_changes : list
        List of :class:`fastmidi.ControlChange` objects.

    '''

    def __init__(self, program, is_drum=False, name=''):
        '''Create the Instrument.

        '''
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.notes = []
        self.pitch_bends = []
        self.control_changes = []

    def get_duration(self):
        '''Returns the total duration of this instrument.

        Returns
        -------
        duration : float
            Duration, in beats, of this instrument.

        '''
        # Cycle through all note ends and all pitch bends and find the largest
        durations = [n.offset+n.duration for n in self.notes]
        # If there are no notes, just return 0
        if len(durations) == 0:
            return 0.
        else:
            return max(durations)

    def get_offsets(self):
        '''Get offsets of all notes played by this instrument.
        May contain duplicates.

        Returns
        -------
        offsets : np.ndarray
                List of all note offsets.

        '''
        offsets = []
        # Get the note-on time of each note played by this instrument
        for note in self.notes:
            offsets.append(note.offset)
        # Return them sorted (because why not?)
        return np.sort(offsets)

    def print(self):
        for n in self.notes:
            print(n)

    def get_piano_roll(self, sample_freq, pedal_threshold=64):
        '''Compute a piano roll matrix of this instrument.

        Parameters
        ----------
        sample_freq : int
            Beat sampling frequency of the timestep, i.e. each timestep is spaced apart
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
            Piano roll of this instrument: 1 - on; 2 - sustained

        '''
        # If there are no notes, return an empty matrix
        if self.notes == []:
            return np.array([[]]*128)
        # Get the end time of the last event
        max_time_step = round(self.get_duration() * sample_freq)

        # Allocate a matrix of zeros - we will add in as we go
        piano_roll = np.zeros((128, max_time_step))
            
        # Add up piano roll matrix, note-by-note
        for n in self.notes:
            offset = round(n.offset*sample_freq)
            duration = round(n.duration*sample_freq)
            piano_roll[n.pitch, offset:offset+duration] += n.velocity
        
        # Process sustain pedals
        if pedal_threshold is not None:
            CC_SUSTAIN_PEDAL = 64
            time_pedal_on = 0
            is_pedal_on = False
            for cc in [_e for _e in self.control_changes
                       if _e.number == CC_SUSTAIN_PEDAL]:
                time_now = round(cc.offset*sample_freq)
                is_current_pedal_on = (cc.value >= pedal_threshold)
                if not is_pedal_on and is_current_pedal_on:
                    time_pedal_on = time_now
                    is_pedal_on = True
                elif is_pedal_on and not is_current_pedal_on:
                    # For each pitch, a sustain pedal 'retains'
                    # the maximum velocity up to now due to
                    # logarithmic nature of human loudness perception
                    subpr = piano_roll[:, time_pedal_on:time_now]

                    # Take the running maximum
                    pedaled = np.maximum.accumulate(subpr, axis=1)
                    piano_roll[:, time_pedal_on:time_now] = pedaled
                    is_pedal_on = False

        #return piano_roll
        
        # Process pitch changes
        # Need to sort the pitch bend list for the following to work
        ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.offset)
        # Add in a bend of 0 at the end of time
        end_bend = PitchBend(0, max_time_step)
        for start_bend, end_bend in zip(ordered_bends,
                                        ordered_bends[1:] + [end_bend]):
            # Piano roll is already generated with everything bend = 0
            if np.abs(start_bend.pitch) < 1:
                continue
            # Get integer and decimal part of bend amount
            start_pitch = pitch_bend_to_semitones(start_bend.pitch)
            bend_int = int(np.sign(start_pitch)*np.floor(np.abs(start_pitch)))
            bend_decimal = np.abs(start_pitch - bend_int)
            # Column indices effected by the bend
            bend_range = np.r_[int(start_bend.offset*sample_freq):int(end_bend.offset*sample_freq)]
            # Construct the bent part of the piano roll
            bent_roll = np.zeros(piano_roll[:, bend_range].shape)
            # Easiest to process differently depending on bend sign
            if start_bend.pitch >= 0:
                # First, pitch shift by the int amount
                if bend_int is not 0:
                    bent_roll[bend_int:] = piano_roll[:-bend_int, bend_range]
                else:
                    bent_roll = piano_roll[:, bend_range]
                # Now, linear interpolate by the decimal place
                bent_roll[1:] = ((1 - bend_decimal)*bent_roll[1:] +
                                 bend_decimal*bent_roll[:-1])
            else:
                # Same procedure as for positive bends
                if bend_int is not 0:
                    bent_roll[:bend_int] = piano_roll[-bend_int:, bend_range]
                else:
                    bent_roll = piano_roll[:, bend_range]
                bent_roll[:-1] = ((1 - bend_decimal)*bent_roll[:-1] +
                                  bend_decimal*bent_roll[1:])
            # Store bent portion back in piano roll
            piano_roll[:, bend_range] = bent_roll

        return piano_roll


    def get_chroma(self, sample_freq, pedal_threshold=64):
        '''Get a sequence of chroma vectors from this instrument.

        Parameters
        ----------
        sample_freq : int
            Beat sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./sample_freq`` beats.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_roll : np.ndarray, shape=(12,timesteps.shape[0])
            Chromagram of this instrument.

        '''
        # First, get the piano roll
        piano_roll = self.get_piano_roll(sample_freq=sample_freq, pedal_threshold=pedal_threshold)
        # Fold into one octave
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def get_pitch_class_histogram(self, use_duration=False, use_velocity=False,
                                  normalize=False):
        '''Computes the frequency of pitch classes of this instrument,
        optionally weighted by their durations or velocities.

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
            Histogram of pitch classes given current instrument, optionally
            weighted by their durations or velocities.
        '''

        # Return all zeros if track is drum
        if self.is_drum:
            return np.zeros(12)

        weights = np.ones(len(self.notes))

        # Assumes that duration and velocity have equal weight
        if use_duration:
            weights *= [note.end - note.start for note in self.notes]
        if use_velocity:
            weights *= [note.velocity for note in self.notes]

        histogram, _ = np.histogram([n.pitch % 12 for n in self.notes],
                                    bins=np.arange(13),
                                    weights=weights,
                                    density=normalize)

        return histogram

    def remove_invalid_notes(self):
        '''Removes any notes whose end time is before or at their start time.

        '''
        # Crete a list of all invalid notes
        notes_to_delete = []
        for note in self.notes:
            if note.duration <= 0:
                notes_to_delete.append(note)
        # Remove the notes found
        for note in notes_to_delete:
            self.notes.remove(note)


    def fluidsynth(self, fs=44100, sf2_path=None):
        '''Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``fastmidi``.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.

        '''
        # If sf2_path is None, use the included TimGM6mb.sf2 path
        if sf2_path is None:
            sf2_path = pkg_resources.resource_filename(__name__, DEFAULT_SF2)

        if not _HAS_FLUIDSYNTH:
            raise ImportError('fluidsynth() was called but pyfluidsynth '
                              'is not installed.')

        if not os.path.exists(sf2_path):
            raise ValueError('No soundfont file found at the supplied path '
                             '{}'.format(sf2_path))

        # If the instrument has no notes, return an empty array
        if len(self.notes) == 0:
            return np.array([])

        # Create fluidsynth instance
        fl = fluidsynth.Synth(samplerate=fs)
        # Load in the soundfont
        sfid = fl.sfload(sf2_path)
        # If this is a drum instrument, use channel 9 and bank 128
        if self.is_drum:
            channel = 9
            # Try to use the supplied program number
            res = fl.program_select(channel, sfid, 128, self.program)
            # If the result is -1, there's no preset with this program number
            if res == -1:
                # So use preset 0
                fl.program_select(channel, sfid, 128, 0)
        # Otherwise just use channel 0
        else:
            channel = 0
            fl.program_select(channel, sfid, 0, self.program)
        # Collect all notes in one list
        event_list = []
        for note in self.notes:
            event_list += [[note.start, 'note on', note.pitch, note.velocity]]
            event_list += [[note.end, 'note off', note.pitch]]
        for bend in self.pitch_bends:
            event_list += [[bend.offset, 'pitch bend', bend.pitch]]
        for control_change in self.control_changes:
            event_list += [[control_change.offset, 'control change',
                            control_change.number, control_change.value]]
        # Sort the event list by time, and secondarily by whether the event
        # is a note off
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        # Add some silence at the beginning according to the time of the first
        # event
        current_time = event_list[0][0]
        # Convert absolute seconds to relative samples
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        # Include 1 second of silence at the end
        event_list[-1][0] = 1.
        # Pre-allocate output array
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros(int(np.ceil(fs*total_time)))
        # Iterate over all events
        for event in event_list:
            # Process events based on type
            if event[1] == 'note on':
                fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off':
                fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                fl.cc(channel, event[2], event[3])
            # Add in these samples
            current_sample = int(fs*current_time)
            end = int(fs*(current_time + event[0]))
            samples = fl.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            # Increment the current sample
            current_time += event[0]
        # Close fluidsynth
        fl.delete()

        return synthesized

    def __repr__(self):
        return 'Instrument(program={}, is_drum={}, name="{}")'.format(
            self.program, self.is_drum, self.name.replace('"', r'\"'))

