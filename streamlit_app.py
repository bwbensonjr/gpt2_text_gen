import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
MODEL = GPT2LMHeadModel.from_pretrained('gpt2')

st.title("Generate text from example with GPT-2")

input_text = st.text_area("Input text")

if input_text:
    inputs = TOKENIZER.encode(input_text, return_tensors='pt')
    outputs = MODEL.generate(inputs, max_length=200, do_sample=True)
    output_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    st.write(output_text)

