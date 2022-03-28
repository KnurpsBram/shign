import numpy as np
import scipy.signal
import librosa

from shign.util import ms_to_samples, samples_to_ms, sec_to_frames, frames_to_ms, audio_to_rms_envelope

def get_shift_ms(audio_a, audio_b, sr_a, sr_b, win_length_ms=25., hop_length_ms=10., min_overlap_sec=1., max_shift_sec=30.):
    """
    Finds the amount of milliseconds the start of `audio_b` must move to the right in order to align it with `audio_a`.

    Audios are compared by their root-mean-square envelopes. See `shign.util.audio_to_rms_envelope()`.

    The RMS envelope consists of one loudness value for every `hop_length_ms` milliseconds worth of audio.
    The optimal alignment is found by shifting the RMS envelopes and recording the similarity between the envelopes for each amount of shift.
    The amount of shift that maximizes the similarity is the optimial amount of shift.
    A low value for `hop_length_ms` is more precise.
    The measure of similarity is the correlation normalized by the amount of valid (non-padding) entries the correlation is taken over.

    Parameters
    ----------
    audio_a : np.ndarray
        Array of waveform values
    audio_b : np.ndarray
        Array of waveform values
    sr_a : int
        Samplerate of `audio_a`
    sr_b : int
        Samplerate of `audio_b`
    win_length_ms : float (optional)
        Window length (milliseconds) that's used to determine the dbfs envelope. If no value is passed, assumes 25 ms.
    hop_length_ms : float
        Hop length (milliseconds) that's used to determine the dbfs envelope. If no value is passed, assumes 10 ms.
    min_overlap_sec : float
        This function assumes the two audios have at least this amount of overlap (in seconds). If no value is passed, assumes 1 second. Passing a too low value can give inaccurate results.
    max_shift_sec : float
        This function assumes the **center** of `audio_b` does not require more than this amount of shift relative to the **center** of `audio_a`. If the value 0 is passed, all shift candidates are considered.

    Returns
    -------
    shift_ms : float
        The amount of milliseconds the **start** of `audio_b` should move to the right to align the audios
    """

    rms_a = audio_to_rms_envelope(audio_a, sr=sr_a, win_length_ms=win_length_ms, hop_length_ms=hop_length_ms)
    rms_b = audio_to_rms_envelope(audio_b, sr=sr_b, win_length_ms=win_length_ms, hop_length_ms=hop_length_ms)

    corr_num   = scipy.signal.correlate(rms_a, rms_b, mode='full')
    corr_denom = scipy.signal.correlate(np.ones_like(rms_a), np.ones_like(rms_b), mode='full')
    corr       = corr_num / corr_denom
    half_idx   = int(np.round(len(corr) / 2))

    if min_overlap_sec != 0:
        n = sec_to_frames(min_overlap_sec, frame_length_ms=hop_length_ms)
        corr[:n]  = 0
        corr[-n:] = 0

    if max_shift_sec != 0:
        n = sec_to_frames(max_shift_sec, frame_length_ms=hop_length_ms)
        corr[:max(half_idx - n, 0)]         = 0
        corr[min(half_idx + n, len(corr)):] = 0

    shift_idx = np.argmax(corr) - half_idx              # this is the amount of frames the **center** of audio_b should move to the right to match audio_a
    shift_idx = shift_idx + (len(rms_a) - len(rms_b))/2 # this is the amount of frames the **start** of audio_b should move to the right to match audio_a
    shift_ms  = frames_to_ms(shift_idx, frame_length_ms=hop_length_ms)

    return shift_ms

def pad_both(audio_a, audio_b, sr_a, sr_b, shift_ms):
    """
    Pads both audios with silence so that they have equal length and are aligned

    Example 1 (psuedocode):
    ```
    audio_a = [1, 2, 3, 4]
    audio_b = [3, 4, 5]
    pad_both(audio_a, audio_b)
    >>> [1, 2, 3, 4, 0]
    >>> [0, 0, 3, 4, 5]
    ```

    Example 2 (psuedocode):
    ```
    audio_a = [1, 2, 3, 4, 5]
    audio_b = [3, 4]
    pad_both(audio_a, audio_b)
    >>> [1, 2, 3, 4, 5]
    >>> [0, 0, 3, 4, 0]
    ```

    Parameters
    ----------
    audio_a : np.ndarray
        Array of waveform values
    audio_b : np.ndarray
        Array of waveform values
    sr_a : int
        Samplerate of `audio_a`
    sr_b : int
        Samplerate of `audio_b`
    shift_ms : float
        The amount of milliseconds the start of `audio_b` should move to the right to align the audios

    Returns
    -------
    audio_a : np.ndarray
        Array of waveform values that may have leading or trailing silence
    audio_b : np.ndarray
        Array of waveform values that may have leading or trailing silence
    """

    # The amount of ms the start of audio_b should shift to the right
    shift_start_ms   = shift_ms
    shift_start_a_ms = -min(0, shift_start_ms)
    shift_start_b_ms = max(0, shift_start_ms)

    shift_start_a_samples = ms_to_samples(shift_start_a_ms, sr=sr_a)
    shift_start_b_samples = ms_to_samples(shift_start_b_ms, sr=sr_b)

    audio_a = np.concatenate([np.zeros(shift_start_a_samples), audio_a], axis=0)
    audio_b = np.concatenate([np.zeros(shift_start_b_samples), audio_b], axis=0)

    # The amount of ms the end of audio_b should move to the right
    shift_end_ms   = samples_to_ms(len(audio_a), sr=sr_a) - samples_to_ms(len(audio_b), sr=sr_b)
    shift_end_a_ms = -min(0, shift_end_ms)
    shift_end_b_ms = max(0, shift_end_ms)

    shift_end_a_samples = ms_to_samples(shift_end_a_ms, sr=sr_a)
    shift_end_b_samples = ms_to_samples(shift_end_b_ms, sr=sr_b)

    audio_a = np.concatenate([audio_a, np.zeros(shift_end_a_samples)], axis=0)
    audio_b = np.concatenate([audio_b, np.zeros(shift_end_b_samples)], axis=0)

    return audio_a, audio_b

def crop_both(audio_a, audio_b, sr_a, sr_b, shift_ms):
    """
    Crops both audios with silence so that they have equal length and are aligned

    Example 1 (psuedocode):
    ```
    audio_a = [1, 2, 3, 4]
    audio_b = [3, 4, 5]
    crop_both(audio_a, audio_b)
    >>> [3, 4]
    >>> [3, 4]
    ```

    Example 2 (psuedocode):
    ```
    audio_a = [1, 2, 3, 4, 5]
    audio_b = [3, 4]
    crop_both(audio_a, audio_b)
    >>> [3, 4]
    >>> [3, 4]
    ```

    Parameters
    ----------
    audio_a : np.ndarray
        Array of waveform values
    audio_b : np.ndarray
        Array of waveform values
    sr_a : int
        Samplerate of `audio_a`
    sr_b : int
        Samplerate of `audio_b`
    shift_ms : float
        The amount of milliseconds the start of `audio_b` should move to the right to align the audios

    Returns
    -------
    audio_a : np.ndarray
        Array of waveform values that may have shorter length than the input
    audio_b : np.ndarray
        Array of waveform values that may have shorter length than the input
    """

    # The amount of ms the start of audio_b should shift to the right
    shift_start_ms   = shift_ms
    shift_start_a_ms = max(0, shift_start_ms)
    shift_start_b_ms = -min(0, shift_start_ms)

    shift_start_a_samples = ms_to_samples(shift_start_a_ms, sr=sr_a)
    shift_start_b_samples = ms_to_samples(shift_start_b_ms, sr=sr_b)

    audio_a = audio_a[shift_start_a_samples:]
    audio_b = audio_b[shift_start_b_samples:]

    # The amount of ms the end of audio_b should move to the right
    shift_end_ms   = samples_to_ms(len(audio_a), sr=sr_a) - samples_to_ms(len(audio_b), sr=sr_b)
    shift_end_a_ms = max(0, shift_end_ms)
    shift_end_b_ms = -min(0, shift_end_ms)

    shift_end_a_samples = ms_to_samples(shift_end_a_ms, sr=sr_a)
    shift_end_b_samples = ms_to_samples(shift_end_b_ms, sr=sr_b)

    if shift_end_a_samples > 0:
        audio_a = audio_a[:-shift_end_a_samples]
    if shift_end_b_samples > 0:
        audio_b = audio_b[:-shift_end_b_samples]

    return audio_a, audio_b

def pad_and_crop_one_to_match_other(audio_a, audio_b, sr_a, sr_b, shift_ms):
    """
    Applies padding and cropping to one audio file so that its start and end match the start and end of the leading audio

    This function only manipulates `audio_a` and does not manipulate `audio_b`

    Example 1 (psuedocode):
    ```
    audio_a = [1, 2, 3, 4]
    audio_b = [3, 4, 5]
    pad_and_crop_one_to_match_other(audio_a, audio_b)
    >>> [3, 4, 0]
    >>> [3, 4, 5]
    ```

    Example 2 (psuedocode):
    ```
    audio_a = [1, 2, 3, 4, 5]
    audio_b = [3, 4]
    pad_and_crop_one_to_match_other(audio_a, audio_b)
    >>> [3, 4]
    >>> [3, 4]
    ```

    Parameters
    ----------
    audio_a : np.ndarray
        Array of waveform values
    audio_b : np.ndarray
        Array of waveform values
    sr_a : int
        Samplerate of `audio_a`
    sr_b : int
        Samplerate of `audio_b`
    shift_ms : float
        The amount of milliseconds the start of `audio_b` should move to the right to align the audios

    Returns
    -------
    audio_a : np.ndarray
        Array of waveform values equal to `audio_a` at input
    audio_b : np.ndarray
        Array of waveform values that may have longer or shorter length than the input and may have leading or trailing silence
    """

    # The amount of ms the start of audio_b should shift to the right
    shift_start_ms      = shift_ms
    shift_start_samples = ms_to_samples(shift_start_ms, sr=sr_b)

    if shift_start_samples < 0:
        audio_b = audio_b[abs(shift_start_samples):]
    else:
        audio_b = np.concatenate([np.zeros(shift_start_samples), audio_b], axis=0)

    # The amount of ms the end of audio_b should move to the right
    shift_end_ms      = samples_to_ms(len(audio_a), sr=sr_a) - samples_to_ms(len(audio_b), sr=sr_b)
    shift_end_samples = ms_to_samples(shift_end_ms, sr=sr_b)

    if shift_end_samples < 0:
        audio_b = audio_b[:-abs(shift_end_samples)]
    else:
        audio_b = np.concatenate([audio_b, np.zeros(shift_end_samples)], axis=0)

    return audio_a, audio_b

def shift_align(audio_a, audio_b, sr_a=None, sr_b=None, align_how="pad_both", min_overlap_sec=1., max_shift_sec=30.):
    """
    Shifts two audios to align them in time.
    The audios must be recordings of the same audio event.

    The amount of shift required to align the audios is found by comparing the root-mean-square envelopes of the audios. See `get_shift_ms()`

    The way audios will be shifted to match each other depends on the parameter `align_how`:

    - `"pad_both"`: both audios will be padded with the necessary amount of silence on the leading and trailing ends. This way no audio will get lost, but aligned audio files may contain a lot of silence.
    - `"crop_both"`: both audios will have their leading and trailing ends cropped to the necessary amount. This only keeps the audio of the event both audios have a recording of. Audio at the tails gets lost.
    - `"pad_and_crop_one_to_match_other"`: only `audio_b` will be modified, either cropped or padded, in order to match `audio_a`. `audio_a` will remain unmodified.

    Parameters
    ----------
    audio_a : np.ndarray or string
        If the input is an array, it will be interpreted as an audio that needs to be aligned to `audio_b`.
        If the input is a string, it will be interpreted as the path to the audio that needs to be loaded.
    audio_b : np.ndarray or string
        If the input is an array, it will be interpreted as an audio that needs to be aligned to `audio_a`.
        If the input is a string, it will be interpreted as the path to the audio that needs to be loaded.
    sr_a : int
        Samplerate of `audio_a`
        If `audio_a` is a string, the audio in that path will be forcibly resampled to `sr_a`. If `sr_a` is `None`, the native samplerate will be used.
    sr_b : int
        Samplerate of `audio_b`
        If `audio_b` is a string, the audio in that path will be forcibly resampled to `sr_b`. If `sr_b` is `None`, the native samplerate will be used.
    align_how : string
        How the audios will be aligned to each other.

    Returns
    -------
    audio_a : np.ndarray
        Array of waveform values
    audio_b : np.ndarray
        Array of waveform values
    """

    if isinstance(audio_a, str):
        audio_a, sr_a = librosa.load(audio_a, sr=sr_a)
    if isinstance(audio_b, str):
        audio_b, sr_b = librosa.load(audio_b, sr=sr_b)

    assert sr_a is not None, "please submit a path to an audio file or an audio array with explicit samplerate"
    assert sr_b is not None, "please submit a path to an audio file or an audio array with explicit samplerate"

    shift_ms = get_shift_ms(audio_a, audio_b, sr_a, sr_b, min_overlap_sec=min_overlap_sec, max_shift_sec=max_shift_sec)

    if   align_how == "pad_both":
        audio_a, audio_b = pad_both(audio_a, audio_b, sr_a, sr_b, shift_ms)
    elif align_how == "crop_both":
        audio_a, audio_b = crop_both(audio_a, audio_b, sr_a, sr_b, shift_ms)
    elif align_how == "pad_and_crop_one_to_match_other":
        audio_a, audio_b = pad_and_crop_one_to_match_other(audio_a, audio_b, sr_a, sr_b, shift_ms)
    else:
        raise Exception(f"Unknown value for align_how {align_how}")

    return audio_a, audio_b
