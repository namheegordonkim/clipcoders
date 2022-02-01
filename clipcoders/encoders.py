from abc import abstractmethod

import clip


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
