'''These classes simply hold MIDI data in a convenient form.

    Ning: Changed offsets and durations in seconds to in beats (quater-notes)
'''
from __future__ import print_function

from .utilities import key_number_to_key_name, bpm_to_tempo, note_number_to_name


class Note(object):
    '''A note event.

    Parameters
    ----------
    pitch : int
        Note pitch, as a MIDI note number.
    offset : float
        Note offset in beats (quarter notes).
    duration : float
        Note duration in beats.
    velocity : int
        Note velocity.
        
    '''

    def __init__(self, pitch, offset, duration, velocity):
        self.pitch = pitch
        self.offset = offset
        self.duration = duration
        self.velocity = velocity

    def __repr__(self):
        return '  <Note> pitch={} ({}), offset={:f}, duration={:f}, velocity={}'.format(
            self.pitch, note_number_to_name(self.pitch), self.offset, self.duration, self.velocity)


class PitchBend(object):
    '''A pitch bend event.

    Parameters
    ----------
    pitch : int
        MIDI pitch bend amount, in the range ``[-8192, 8191]``.
    offset : float
        offset in beats where the pitch bend occurs.

    '''

    def __init__(self, pitch, offset):
        self.pitch = pitch
        self.offset = offset

    def __repr__(self):
        return '  <PitchBend> pitch={:d}, offset={:f}'.format(self.pitch, self.offset)


class ControlChange(object):
    '''A control change event.

    Parameters
    ----------
    number : int
        The control change number, in ``[0, 127]``.
    value : int
        The value of the control change, in ``[0, 127]``.
    offset : float
        offset in beats where the control change occurs.

    '''

    def __init__(self, number, value, offset):
        self.number = number
        self.value = value
        self.offset = offset

    def __repr__(self):
        return ('  <ControlChange> number={:d}, value={:d}, '
                'offset={:f}'.format(self.number, self.value, self.offset))


class TimeSignature(object):
    '''Container for a Time Signature event, which contains the time signature
    numerator, denominator and the event offset in beats.

    Attributes
    ----------
    numerator : int
        Numerator of time signature.
    denominator : int
        Denominator of time signature.
    offset : float
        offset of event in beats.

    Examples
    --------
    Instantiate a TimeSignature object with 6/8 time signature at 4.5 beats:

    >>> ts = TimeSignature(6, 8, 4.5)
    >>> print(ts)
    6/8 at 4.5 beats

    '''

    def __init__(self, numerator, denominator, offset):
        if not (isinstance(numerator, int) and numerator > 0):
            raise ValueError(
                '{} is not a valid `numerator` type or value'.format(
                    numerator))
        if not (isinstance(denominator, int) and denominator > 0):
            raise ValueError(
                '{} is not a valid `denominator` type or value'.format(
                    denominator))
        if not (isinstance(offset, (int, float)) and offset >= 0):
            raise ValueError(
                '{} is not a valid `offset` type or value'.format(offset))

        self.numerator = numerator
        self.denominator = denominator
        self.offset = offset

    def __repr__(self):
        return '  <TimeSignature> numerator={}, denominator={}, offset={}'.format(
            self.numerator, self.denominator, self.offset)

    def __str__(self):
        return '{}/{} at {:.2f} beats'.format(
            self.numerator, self.denominator, self.offset)


class KeySignature(object):
    '''Contains the key signature and the event offset in beats.
    Only supports major and minor keys.

    Attributes
    ----------
    key_number : int
        Key number according to ``[0, 11]`` Major, ``[12, 23]`` minor.
        For example, 0 is C Major, 12 is C minor.
    offset : float
        offset of event in beats.

    Examples
    --------
    Instantiate a C# minor KeySignature object at 4.5 beats:

    >>> ks = KeySignature(13, 4.5)
    >>> print(ks)
    C# minor at 4.5 beats
    '''

    def __init__(self, key_number, offset):
        if not all((isinstance(key_number, int),
                    key_number >= 0,
                    key_number < 24)):
            raise ValueError(
                '{} is not a valid `key_number` type or value'.format(
                    key_number))
        if not (isinstance(offset, (int, float)) and offset >= 0):
            raise ValueError(
                '{} is not a valid `offset` type or value'.format(offset))

        self.key_number = key_number
        self.offset = offset

    def __repr__(self):
        return '  <KeySignature> key_number={}, offset={}'.format(
            self.key_number, self.offset)

    def __str__(self):
        return '{} at {:.2f} beats'.format(
            key_number_to_key_name(self.key_number), self.offset)


class Lyric(object):
    '''Timestamped lyric text.

    Attributes
    ----------
    text : str
        The text of the lyric.
    offset : float
        The offset in beats of the lyric.
    '''

    def __init__(self, text, offset):
        self.text = text
        self.offset = offset

    def __repr__(self):
        return 'Lyric(text="{}", offset={})'.format(
            self.text.replace('"', r'\"'), self.offset)

    def __str__(self):
        return '"{}" at {:.2f} beats'.format(self.text, self.offset)

class Tempo(object):
    '''Contains the tempo and the event offset in beats.

    Attributes
    ----------
    bpm : int
        beats per minute
    tempo : int
        microseconds per beat
    offset : float
        The offset in beats of the event.
    '''

    def __init__(self, bpm, offset):
        self.bpm = int(bpm)
        self.tempo = bpm_to_tempo(bpm) # MIDI tempo
        self.offset = offset

    def __repr__(self):
        return '  <Tempo> bpm="{}", offset={}'.format(
            self.bpm, self.offset)

    def __str__(self):
        return '{} at {:.2f} beats'.format(self.bpm, self.offset)
