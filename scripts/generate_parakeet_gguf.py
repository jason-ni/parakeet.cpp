import mlx.core as mx
import os


def save_model(model_dir):
    # download the model from https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2
    model_file = os.path.join(model_dir, "model.safetensors")
    model = mx.load(model_file)
    mx.save_gguf("parakeet-tdt-0.6b-v2-float32.gguf", model)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="path to the audio file")
    args = parser.parse_args()
    model_dir = args.model_dir
    print(f"Generating GGUF for {model_dir}")
    save_model(model_dir)