from abc import abstractmethod

from torchvision.transforms import Normalize

import clip
import torch

from clipcoders.modules import MakeCutouts


class TextEncoder:
    """
    A generic abstract class encapsulating the behaviour of a text encoder.
    Takes a text as an input and produces an embedding corresponding to the text,
    based on latents learned via a pre-trained encoder.
    """

    @abstractmethod
    def encode(self, text: str):
        """
        Produce an embedding corresponding to the input text.

        :param text: English-language input to be encoded.
        :return: embedding: an embedding corresponding to the input text.
        """
        raise NotImplementedError


class CLIPTextEncoder(TextEncoder):

    def __init__(self, clip_model, device):
        self.clip_model = clip_model
        self.device = device

    def encode(self, text: str):
        # Tokenize and use CLIP's provided logic and parameters to produce the embedding
        return self.clip_model.encode_text(clip.tokenize(text).to(self.device)).float()


class ImageEncoder:

    @abstractmethod
    def encode(self, image: torch.tensor):
        raise NotImplementedError


class CLIPImageEncoder(ImageEncoder):
    def __init__(self, clip_model, device, cutn, cut_pow):
        self.clip_model = clip_model
        self.device = device
        self.cutn = cutn
        self.cut_pow = cut_pow

    def encode(self, image: torch.tensor):
        normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711])
        make_cutouts = MakeCutouts(self.clip_model.visual.input_resolution, self.cutn, self.cut_pow)
        clip_in = normalize(make_cutouts((image + 1) / 2))
        # image_embeds = self.clip_model.encode_image(clip_in).view([self.cutn, image.shape[0], -1])
        image_embeds = self.clip_model.encode_image(clip_in).view([image.shape[0], self.cutn, -1])
        return image_embeds
