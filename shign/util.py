import numpy as np

def frames_to_ms(frames, frame_length_ms):
    return frames * frame_length_ms

def frames_to_sec(frames, frame_length_ms):
    return frames_to_ms(frames, frame_length_ms/1000)

def ms_to_frames(ms, frame_length_ms):
    return int(ms / frame_length_ms)

def ms_to_samples(ms, sr):
    return sec_to_samples(ms/1000, sr=sr)

def samples_to_sec(samples, sr):
    return samples / sr

def samples_to_ms(samples, sr):
    return 1000 * samples_to_sec(samples, sr=sr)

def sec_to_samples(sec, sr):
    return int(sec * sr)

def sec_to_frames(sec, frame_length_ms):
    return ms_to_frames(1000 * sec, frame_length_ms)

def audio_to_rms_envelope(audio, sr=16000, win_length_ms=25, hop_length_ms=10):
    """
    Obtain the Root-Mean-Square envelope of the audio

    The RMS envelope is related to the loudness envelope of an audio waveform

    We define the envelope as the RMS over 25-millisecond frames taken 10 milliseconds apart (the frames overlap).

    Parameters
    ----------
    audio : np.ndarray
        Amplitude values of the waveform (bounded by -1 and 1)
    sr : int (optional)
        Samplerate of the audio. Default is 16 kHz.
    win_length_ms : float (optional)
        Length of the windows the RMS is taken over, in milliseconds. Default is 25 milliseconds.
    hop_length_ms : float (optional)
        Step size between windows. Default is 10 milliseconds.
    Returns
    -------
    rms_envelope : np.ndarray
        RMS values of the waveform
    """

    win_length = ms_to_samples(win_length_ms, sr=sr)
    hop_length = ms_to_samples(hop_length_ms, sr=sr)

    rms_envelope = np.array([np.sqrt(np.mean(audio[i:i+win_length]**2)) for i in range(0, audio.shape[-1] - win_length, hop_length)])

    return rms_envelope
