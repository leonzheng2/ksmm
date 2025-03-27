import transformers
import torch


pipeline = transformers.pipeline(
    "text-generation", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

pipeline("Hey how are you doing today?")
