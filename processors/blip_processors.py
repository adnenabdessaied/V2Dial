"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import torch
from processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)


        segment_mean = (0.485, 0.456, 0.406)
        segment_std = (0.229, 0.224, 0.225)

        self.normalize = transforms.Normalize(segment_mean, segment_std)


class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#|:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


class BlipDialogProcessor(BlipCaptionProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def pre_caption_rm_period(self, text):
        text = re.sub(
            r"([.!\"()*#|:;~])",
            " ",
            text.lower(),
        )
        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        )
        text = text.rstrip("\n")
        text = text.strip(" ")

        # truncate caption
        text_words = text.split(" ")
        if len(text_words) > self.max_words:
            text = " ".join(text_words[: self.max_words])
        return text

    def pre_caption(self, text):
        text = re.sub(
            r"([\"()*#|:;~])",
            " ",
            text.lower(),
        )
        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        )
        text = text.rstrip("\n")
        text = text.strip(" ")

        # truncate caption
        text_words = text.split(" ")
        if len(text_words) > self.max_words:
            text = " ".join(text_words[: self.max_words])
        return text

    def __call__(self, caption, remove_period=False):
        if remove_period:
            caption = self.prompt + self.pre_caption_rm_period(caption)
        else:
            caption = self.prompt + self.pre_caption(caption)
        return caption


class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        # self.transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             image_size,
        #             scale=(min_scale, max_scale),
        #             interpolation=InterpolationMode.BICUBIC,
        #         ),
        #         transforms.ToTensor(),
        #         self.normalize,
        #     ]
        # )
        self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )



    # ### segment anything
    # '''
    #         x = (x - self.pixel_mean) / self.pixel_std

    #     # Pad
    #     h, w = x.shape[-2:]
    #     padh = self.image_encoder.img_size - h
    #     padw = self.image_encoder.img_size - w
    #     x = F.pad(x, (0, padw, 0, padh))
    # '''

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)