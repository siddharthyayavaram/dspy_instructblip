import dspy
from PIL import Image

from dsp.modules.instructblip import instructblip

model_name = "Salesforce/instructblip-flan-t5-xl"

ib = instructblip(model_name)
dspy.settings.configure(lm=ib)

image_path = "2.png"
print("RUNNING.....\n")

p1 = ib(prompt="Describe the image in detail",image_path = image_path)
print(p1)

print("DONE!")