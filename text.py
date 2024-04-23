from transformers import pipeline
pipe = pipeline("text-generation", model="./hf_hub/models--internlm--internlm-chat-7b")