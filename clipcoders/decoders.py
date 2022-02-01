from abc import abstractmethod
from functools import partial

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from clipcoders.diffusion import utils, sampling


class MakeCutouts(nn.Module):
    """
    Helper class that operates on PyTorch tensor-version of images to produce cutouts of images
    """

    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


class ImageDecoder:
    """
    An abstract class encapsulating the generic behaviour of an image decoder.
    An image decoder takes an embedding as an input and outputs a corresponding image.
    """

    @abstractmethod
    def decode(self, embedding: torch.tensor):
        """
        Decode the latent embedding into a corresponding image.
        :param embedding: a torch tensor to be rendered into an image.
        :return: a 3D torch image tensor
        """
        raise NotImplementedError


class CLIPGuidedDiffusionDecoder(ImageDecoder):
    """
    A diffusion-based ImageDecoder that uses a pre-trained CLIP image encoder as a guide
    """

    def __init__(self, clip_model, diffusion_model, device, cutn, cut_pow, clip_guidance_scale, n_steps):
        self.n_steps = n_steps
        self.diffusion_model = diffusion_model
        self.clip_guidance_scale = clip_guidance_scale
        self.clip_model = clip_model
        self.device = device
        self.cutn = cutn
        self.cut_pow = cut_pow

    def decode(self, embedding: torch.tensor):
        make_cutouts = MakeCutouts(self.clip_model.visual.input_resolution, self.cutn, self.cut_pow)
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        def cond_fn(x, t, pred, clip_embed):
            clip_in = normalize(make_cutouts((pred + 1) / 2))
            image_embeds = self.clip_model.encode_image(clip_in).view([self.cutn, x.shape[0], -1])
            losses = spherical_dist_loss(image_embeds, clip_embed[None])
            loss = losses.mean(0).sum() * self.clip_guidance_scale
            grad = -torch.autograd.grad(loss, x)[0]
            return grad

        if hasattr(self.diffusion_model, 'clip_model'):
            extra_args = {'clip_embed': embedding}
            cond_fn_ = cond_fn
        else:
            extra_args = {}
            cond_fn_ = partial(cond_fn, clip_embed=embedding)
        _, side_y, side_x = self.diffusion_model.shape

        x = torch.randn([1, 3, side_y, side_x], device=self.device)
        t = torch.linspace(1, 0, self.n_steps + 1, device=self.device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)

        return sampling.cond_sample(
            self.diffusion_model,
            x,
            steps,
            0,
            extra_args,
            cond_fn_
        )


class ClassifierFreeGuidanceDecoder(ImageDecoder):
    """
    A diffusion-based ImageDecoder that doesn't explicitly require an image encoder.
    """

    def __init__(self, diffusion_model, device, n_steps):
        self.n_steps = n_steps
        self.diffusion_model = diffusion_model
        self.device = device

    def decode(self, embedding: torch.tensor):
        zero_embed = torch.zeros_like(embedding)
        target_embeds = [zero_embed, embedding]
        weights = torch.tensor([-4., 5.], device=self.device)

        def cfg_model_fn(x, t):
            n = x.shape[0]
            n_conds = len(target_embeds)
            x_in = x.repeat([n_conds, 1, 1, 1])
            t_in = t.repeat([n_conds])
            clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
            vs = self.diffusion_model.forward(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
            v = vs.mul(weights[:, None, None, None, None]).sum(0)
            return v

        _, side_y, side_x = self.diffusion_model.shape

        x = torch.randn([1, 3, side_y, side_x], device=self.device)
        t = torch.linspace(1, 0, self.n_steps + 1, device=self.device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)

        return sampling.sample(
            cfg_model_fn,
            x,
            steps,
            1,
            {}
        )


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])
