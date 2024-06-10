from typing import Any, Union
import numpy as np
from dsp.modules.vlm import VLM
from dspy.primitives.vision import Image, SupportsImage
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image

# model_main = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
# processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
# device = "cuda"
# model_main.to(device)

class instructblip(VLM):
    def __init__(self, model_name):

        model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
        processor = InstructBlipProcessor.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        self.device = device
        self.model = model
        self.processor = processor
        self.provider = "default"
        self.history = []
        self.kwargs = {
            "do_sample": True,
            "num_beams": 5,
            "max_length": 1000,
            "min_length": 1,
            "top_p": 0.9,
            "repetition_penalty": 2.0,
            "length_penalty": 1.0,
            "temperature": 1,
        }

    def basic_request(self, prompt: str, image, **kwargs) -> Any:
        # print(prompt)
        merged_kwargs = {**self.kwargs, **kwargs}

        # if image is not None:
        #     image = Image(image) if not isinstance(image, Image) else image

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # print("\n")
        # print(merged_kwargs)

        outputs = self.model.generate(
            **inputs,
            **merged_kwargs
        )

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        # print(response)

        self.history.append({
            "prompt": prompt,
            "image" : image,
            "response": response,
            "kwargs": merged_kwargs,
        })
        return response

    def __call__(self, prompt, image_path, only_completed=True, return_sorted=False, **kwargs):
        image_act = Image.open(image_path).convert("RGB")
        response = self.basic_request(prompt, image_act, **kwargs)
        return response