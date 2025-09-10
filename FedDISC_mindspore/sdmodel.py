from dataclasses import dataclass
import numpy as np
from mindspore import nn
import argparse
import PIL
import mindspore
from mindone.diffusers.pipelines import StableDiffusionImg2ImgPipeline,StableDiffusionPipeline
from mindone.diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from typing import Callable, List, Optional, Union
from mindone.diffusers.utils import BaseOutput
from mindone.diffusers.models.embeddings import get_timestep_embedding

has_cuda = True

device = 'cpu' if not has_cuda else 'cuda'
sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", mindspore_dtype=mindspore.float16).to(device)


pipe = sd_pipe

safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", mindspore_dtype=mindspore.float16)
safety_checker = safety_checker.to(device)

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["mindspore.Generator"], "mindspore.Generator"]] = None,
    device: Optional["mindspore.device"] = None,
    dtype: Optional["mindspore.dtype"] = None,
    layout: Optional["mindspore.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or mindspore.strided
    device = device or mindspore.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            mindspore.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = mindspore.cat(latents, dim=0).to(device)
    else:
        latents = mindspore.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]

class ClientImageEncoder():
    def __init__(
        self
    ):
        self.device = device
        
    def _encode_image(self, image, device, batch_size, noise_level, generator,return_1024=False):
        dtype = next(sd_pipe.image_encoder.parameters()).dtype
        
        image = mindspore.tensor(image.clone().detach(),dtype = mindspore.float16)
        
        image_embeds = sd_pipe.image_encoder(image).image_embeds
        if return_1024 ==True:
            return image_embeds
        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )
        
        return image_embeds
    
    def noise_image_embeddings(
        self,
        image_embeds: mindspore.Tensor,
        noise_level: int,
        noise: Optional[mindspore.FloatTensor] = None,
        generator: Optional[mindspore.Generator] = None,
        no_concat=False
    ):
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )
        noise_level = mindspore.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        image_embeds = sd_pipe.image_normalizer.scale(image_embeds)

        image_embeds = sd_pipe.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        image_embeds = sd_pipe.image_normalizer.unscale(image_embeds)
        if no_concat == True:
            return image_embeds
        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(image_embeds.dtype)
        image_embeds = mindspore.cat((image_embeds, noise_level), 1)
        return image_embeds
    
    @mindspore.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], mindspore.FloatTensor],
        noise_level: int = 0,
        generator: Optional[mindspore.Generator] = None,
        return_1024 = False
    
    ):
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        
        batch_size = image.shape[0]

        noise_level = mindspore.tensor([noise_level], device=device)
        # 3. Encode input image
        image_embeddings = self._encode_image(image, self.device, batch_size, noise_level,generator,return_1024)
        
        return image_embeddings
    
    
class ImageGenerator():
    def __init__(
        self
    ):
        self.device = device
        
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * pipe.scheduler.init_noise_sigma
        return latents  
    
    def noise_image_embeddings(
        self,
        image_embeds: mindspore.Tensor,
        noise_level: int,
        noise: Optional[mindspore.FloatTensor] = None,
        generator: Optional[mindspore.Generator] = None,
        no_concat=False
    ):
        
        if image_embeds.shape[1] == 2048:
            imfea = image_embeds[0][:1024].unsqueeze(0)
            timefea = image_embeds[0][1024:].unsqueeze(0)
            image_embeds = imfea
            
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )
        noise_level = mindspore.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        image_embeds = sd_pipe.image_normalizer.scale(image_embeds)

        image_embeds = sd_pipe.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        image_embeds = sd_pipe.image_normalizer.unscale(image_embeds)
        if no_concat == True:
            return image_embeds
        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(image_embeds.dtype)
        noise_level = (noise_level+timefea)/2
        
        image_embeds = mindspore.cat((image_embeds, noise_level), 1)
        return image_embeds
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    @mindspore.no_grad()
    def __call__(
        self,
        image_embeddings: mindspore.FloatTensor,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        global_embeddings: Optional[mindspore.FloatTensor] = None,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 2,
        num_images_per_prompt: Optional[int] = 1,
        prompt_embeds: Optional[mindspore.FloatTensor] = None,
        negative_prompt_embeds: Optional[mindspore.FloatTensor] = None,
        generator: Optional[Union[mindspore.Generator, List[mindspore.Generator]]] = None,
        latents: Optional[mindspore.FloatTensor] = None,
        eta: float = 0.0,
        callback: Optional[Callable[[int, int, mindspore.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        mindspore.cuda.empty_cache()
        # 0. concat public feature
        #image_embeddings：2*1*768，image_embeddings[0]=uncond image_embeddings[1] = img_embedding
        #public_embeddings： 1*1*768
        if prompt is None and prompt_embeds is None:
            prompt = ""
        
        if global_embeddings!=None:
            image_embeddings = mindspore.cat([image_embeddings,global_embeddings.unsqueeze(0)],dim=0)
            #image_embeddings = 0.5*image_embeddings + 0.5*public_embeddings
        
        # 1. Default height and width to unet
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        
        # 2. Define guidance and batchsize
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1
        
        # 3. Set timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = pipe.scheduler.timesteps
        
        # 4. Prepare latent variables
        
        prompt_embeds = pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        if global_embeddings!=None:
            prompt_embeds = mindspore.cat([prompt_embeds,prompt_embeds[1].unsqueeze(0)],dim=0)
            
            
        num_channels_latents = pipe.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        # with sd_pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            
            #latent_model_input = mindspore.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            latent_model_input = mindspore.cat([latents] * 3) if global_embeddings!=None else mindspore.cat([latents] * 2)
            
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,class_labels=image_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                
                if global_embeddings!=None:
                    noise_pred_uncond, noise_pred_img,noise_pred_glo= noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_img - noise_pred_uncond) + guidance_scale * (noise_pred_glo - noise_pred_uncond)        
                else:
                
                    noise_pred_uncond, noise_pred_img= noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_img - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        image = pipe.decode_latents(latents)
        
        safety_checker_input = pipe.feature_extractor(pipe.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, nsfw_content_detected = safety_checker(
            images=image, clip_input=safety_checker_input.pixel_values.to(prompt_embeds.dtype)
        )
            
        # nsfw_content_detected = None
        image = pipe.numpy_to_pil(image)
        
        return StableDiffusionPipelineOutput(images=image,nsfw_content_detected=nsfw_content_detected)