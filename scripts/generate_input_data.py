import mlx.core as mx
from parakeet_mlx import from_pretrained
from parakeet_mlx.parakeet import load_audio, get_logmel
from parakeet_mlx.audio import PreprocessArgs
import numpy as np


def generate_input_data(audio_path,  output_path):
    # Load the audio
    audio = load_audio(audio_path, 16000)

    # Compute the log-mel spectrogram
    preprocess_args = PreprocessArgs(
        sample_rate=16000,
        normalize='per_feature',
        window_size=0.025,
        window_stride=0.01,
        window='hann',
        features=128,
        n_fft=512,
        dither=1e-5,
        pad_to=0,
        pad_value=0.0,
    )

    logmel = get_logmel(audio, preprocess_args)

    # save the input data in raw format
    data = np.array(logmel).tobytes()
    with open(output_path, 'wb') as f:
        f.write(data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="path to the audio file")
    parser.add_argument("output_path", type=str, help="path to the output file")
    args = parser.parse_args()
    generate_input_data(args.audio_path, args.output_path)