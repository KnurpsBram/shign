# Shign
**Shi**ft and al**ign** two recordings of the same audio event. The alignment is found by as the amount of shift that maximizes the correlation between loudness envelopes of the audios.

> Shign on, you crazy diamond

You can obtain aligned audios by either
- padding both audios with silence
- cropping both audios where there is no overlap
- cropping and padding one audio to match the start and end of the leading audio

This package cannot align two recordings of separate audio events, such as two recordings of separate instances where the same sentence is spoken. If that's what you're looking for, you're looking for [Dynamic Time Warping](https://librosa.org/doc/main/generated/librosa.sequence.dtw.html).

If you have two recordings of the same audio event that started and stopped recording at different times, you're in the right place.

### Usage
For detailed usage description, see the [examples](examples) or the [Shign Documentation](https://knurpsbram.github.io/shign/shign.html)

Use the python script to align two audios (you need to have the package installed and this repo cloned on your local machine)
```
python shift_align.py -i my_file1.wav my_file2.wav -o my_aligned_file1.wav my_aligned_file2.wav --align_how crop_both
```

Import the package into a different project (you need to clone this repo only once to install it, you can delete it after)
```
from shign import shift_align

my_aligned_audio1, my_aligned_audio2 = shift_align(path1="my_file1.wav", path2="my_file2.wav", align_how="crop_both")
```

### Installation
```
git clone https://github.com/KnurpsBram/shign
cd shign
pip install .
```
Remove the repo after installation (optional)
```
cd ../
rm -rf shign
```

### Alternatives
If you're here, you may want to check these out too:  
- https://github.com/benfmiller/audalign
- https://github.com/allisonnicoledeal/VideoSync
