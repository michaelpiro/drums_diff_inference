import os
import random
from audio.audio_processing import preprocess_for_inference, SAMPLE_RATE
import numpy as np
import torch

import diffusers
from diffusers import AudioLDM2Pipeline
from diffusers import AudioLDM2UNet2DConditionModel
import torchaudio
from diffusers import AudioLDM2Pipeline
from diffusers.utils.torch_utils import randn_tensor
import os
import scipy


def get_pipe():
    audio_ldm_repo_id = "cvssp/audioldm2-music"
    pipe = AudioLDM2Pipeline.from_pretrained(audio_ldm_repo_id, torch_dtype=torch.float32)
    return pipe

def load_my_unet():
    my_repo_id = "michaelpiro1/new_unet_fromLDM"
    my_unet = AudioLDM2UNet2DConditionModel.from_pretrained(my_repo_id, subfolder="unet", torch_dtype=torch.float16)
    return my_unet

def load_audio_and_prompt(base_dir):
    data = []

    for subdir, _, files in os.walk(base_dir):
        drums_file = None
        no_drums_file = None
        prompt_file = None
        original_file = None
        for file in files:
            if file == 'drums.mp3':
                drums_file = os.path.join(subdir, file)
            elif file == "no_drums.mp3":
                no_drums_file = os.path.join(subdir, file)
            elif file.endswith('.txt'):
                prompt_file = os.path.join(subdir, file)
            elif file == 'original.mp3':
                original_file = os.path.join(subdir, file)
        if drums_file and no_drums_file and prompt_file:
            with open(prompt_file, 'r') as txt_file:
                prompt = txt_file.read()

            data.append({
                'drums_audio': drums_file,
                'no_drums_audio': no_drums_file,
                'prompt': prompt,
                'path': subdir,
                'original': original_file

            })

    return data

# # Example usage
# dir = "/content/drive/MyDrive/survey"
# base_dir = os.path.join(dir, 'audio')
# data = load_audio_and_prompt(base_dir)
# for item in data:
#     print(f"Drums Audio: {item['drums_audio']}")
#     print(f"No Drums Audio: {item['no_drums_audio']}")
#     print(f"Prompt: {item['prompt']}")
#     print(f"Path: {item['path']}")
#     print(f"original: {item['original']}\n")


# from IPython.display import Audio

@torch.no_grad()
def audio_ldm2_inference(
        pipe: AudioLDM2Pipeline,
        no_drums_path: str,
        output_path: str,
        prompt: str,
        audio_length_in_s: float = 7.1,
        num_inference_steps: int = 100,
        callback=None,
        callback_steps: int = 1000,
        generator=None,
        guidance_scale: float = 3.5,
        eta: float = 0.0,
        dtype=torch.float32,
        alpha: float = 0.0,

):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    pipe = pipe.to(device)

    # load the audio
    audio, sr = torchaudio.load(no_drums_path)

    # extract the mel spectrogram
    mel = preprocess_for_inference(audio, sr).to(dtype).to(device)

    # encoding the spectogram to latent space
    if device == "cuda":
        latents = pipe.vae.encode(mel).latent_dist.sample()
    else:
        latents = torch.randn((4,4,64,256), device=device, dtype=dtype)
    # latent = pipe.vae.encode(mel).latent_dist.mean

    # scaling the latent
    latents = pipe.vae.config.scaling_factor * latents
    original_latents = latents

    if prompt is None:
        prompt = "Drums"

    prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        do_classifier_free_guidance=True,
        num_waveforms_per_prompt=1,
    )

    # generate the audio
    audio = pipe(
        # prompt=None,
        latents=latents,
        num_inference_steps=num_inference_steps,
        guidance_scale = guidance_scale,
        eta=eta,
        callback=callback,
        callback_steps=callback_steps,
        generator=generator,
        output_type = "np",
        audio_length_in_s=audio_length_in_s,
        prompt_embeds=prompt_embeds,
        generated_prompt_embeds=generated_prompt_embeds,
        attention_mask=attention_mask,

    ).audios
    # save the audio
    save_file_path = os.path.join(output_path, f"DrumsDiff_.wav")
    scipy.io.wavfile.write(save_file_path, rate=16000, data=audio[0])

    return audio, original_latents



@torch.no_grad()
def drum_diff_inference(
        pipe: AudioLDM2Pipeline,
        no_drums_path: str,
        output_path: str,
        prompt: str,
        audio_length_in_s: float = 7.1,
        num_inference_steps: int = 100,
        callback=None,
        callback_steps: int = 1000,
        generator=None,
        guidance_scale: float = 3.5,
        eta: float = 0.0,
        dtype=torch.float16,
        alpha: float = 0.0,

):
    do_classifier_free_guidance = guidance_scale > 1.0
    beta = 1 - alpha

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    pipe = pipe.to(device)
    pipe.eval()

    # load the audio
    audio, sr = torchaudio.load(no_drums_path)

    # extract the mel spectrogram
    mel = preprocess_for_inference(audio, sr).to(dtype).to(device)

    # encoding the spectogram to latent space
    latents = pipe.vae.encode(mel).latent_dist.sample()
    # latent = pipe.vae.encode(mel).latent_dist.mean

    # scaling the latent
    latents = pipe.vae.config.scaling_factor * latents
    original_latents = latents

    # unscaled_latent = latent
    # latent = latent / pipe.scheduler.init_noise_sigma

    vocoder_upsample_factor = np.prod(
        pipe.vocoder.config.upsample_rates) / pipe.vocoder.config.sampling_rate

    if audio_length_in_s is None:
        audio_length_in_s = pipe.unet.config.sample_size * pipe.vae_scale_factor * vocoder_upsample_factor

    # height = int(audio_length_in_s / vocoder_upsample_factor)

    # original_waveform_length = int(audio_length_in_s * pipe.vocoder.config.sampling_rate)
    # if height % pipe.vae_scale_factor != 0:
    #     height = int(np.ceil(height / pipe.vae_scale_factor)) * pipe.vae_scale_factor
        # logger.info(
        #     f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
        #     f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
        #     f"denoising process."
        # )

    # encoding the prompt
    prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        1,
        False,
        # negative_prompt = negative_prompt,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timesteps = torch.arange(num_inference_steps, 0, -1).to(device) * 10
    timesteps = torch.cat([torch.tensor([400]), torch.arange(num_inference_steps, 0, -1) * 10]).to(device)

    # 5. Prepare latent variables
    # num_channels_latents = pipe.unet.config.in_channels*2
    noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)

    # scale the initial noise by the standard deviation required by the scheduler
    # noise = noise * pipe.scheduler.init_noise_sigma

    # noisy_latent = pipe.scheduler.add_noise(torch.zeros_like(noise), noise,
    #                                         torch.tensor([timesteps[0]]).to(device))

    noisy_latents = pipe.scheduler.add_noise(latents, noise, torch.tensor([timesteps[0]]).to(device))

    # noisy_latent = noise
    # noisy_latent = noise * pipe.scheduler.init_noise_sigma
    # noisy_latent = noise

    # 6. Prepare extra step kwargs
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # alpha = 0.15
            # beta = 1 - alpha
            noisy_latents = beta * noisy_latents + alpha * original_latents
            noisy_latents = pipe.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    pipe.maybe_free_model_hooks()

    # 8. Post-processing

    noisy_latents = 1 / pipe.vae.config.scaling_factor * noisy_latents

    mel_spectrogram = pipe.vae.decode(noisy_latents).sample

    output = pipe.mel_spectrogram_to_waveform(mel_spectrogram)
    save_file_path = os.path.join(output_path, f"DrumsDiff_.wav")
    # scipy.io.wavfile.write(save_dir, rate=16000, data=audio)

    torchaudio.save(save_file_path, output, SAMPLE_RATE)
    # audio = audio[:, :original_waveform_length]

    # torchaudio.save("/content/drive/MyDrive/survey/great results/awsome.mp3", output, sample_rate = 16000)