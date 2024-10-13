from utils.utils import audio_ldm2_inference, get_pipe, drum_diff_inference, load_my_unet
import torch
import argparse

NO_DRUMS_PATH = "/Users/mac/Desktop/demucs_out/mdx_extra/150073/no_drums.mp3"
OUTPUT_PATH = "output.wav"

def parse_args():
    parser = argparse.ArgumentParser()

    # 1. Define the arguments
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=3)
    parser.add_argument("--audio_length_in_s", type=float, default=7.1)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--no_drums_path", type=str, default=NO_DRUMS_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--unet", type=str, default=None)

    # 2. Parse the arguments
    parser = parser.parse_args()

    # 3. Return the parsed arguments
    return parser

def main():
    args = parse_args()

    # 4. Load the model
    pipe = get_pipe()
    if args.unet == "drum_diff":
        u_net = load_my_unet()
        pipe.unet = u_net


    if args.dtype == "torch.float16":
        # pipe = pipe.half()
        dtype = torch.float16
    else:
        # pipe = pipe.float()
        dtype = torch.float32



    # 5. Run the inference
    if args.unet == "drum_diff":
        print("running Drum Diff Inference...")
        drum_diff_inference(pipe,
                            args.no_drums_path,
                            args.output_path,
                            args.prompt,
                            args.audio_length_in_s,
                            args.num_inference_steps,
                            None,
                            1000,
                            None,
                            args.guidance_scale,
                            args.eta,
                            dtype,
                            args.alpha,
                            )
    else:
        print("running Audio LDM2 Inference...")
        audio_ldm2_inference(pipe,
                             args.no_drums_path,
                             args.output_path,
                             args.prompt,
                             args.audio_length_in_s,
                             args.num_inference_steps,
                             None,
                             1000,
                             None,
                             args.guidance_scale,
                             args.eta,
                             dtype,
                             args.alpha,
                             )


if __name__ == "__main__":
    main()

