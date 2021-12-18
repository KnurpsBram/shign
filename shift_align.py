import argparse
import soundfile as sf

import shign

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", type=str, nargs=2)
    parser.add_argument("-o", type=str, nargs=2)
    parser.add_argument("--align_how", type=str, default="pad_both")
    parser.add_argument("--min_overlap_sec", type=float, default=1.)
    parser.add_argument("--max_shift_sec", type=float, default=30.)

    args = vars(parser.parse_args())

    audio_a, audio_b = shign.shift_align(args["i"][0], args["i"][1], align_how=args["align_how"], min_overlap_sec=args["min_overlap_sec"], max_shift_sec=args["max_shift_sec"])

    sf.write(data=audio_a, file=args["o"][0], samplerate=sf.info(args["i"][0]).samplerate)
    sf.write(data=audio_b, file=args["o"][1], samplerate=sf.info(args["i"][1]).samplerate)
