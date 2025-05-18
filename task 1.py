import torch
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the saved fine-tuned model
model_path = "./bart-xsum-small/checkpoint-300"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Summary function
def generate_summary(text, word_count=64):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    target_token_count = int(word_count * 1.3)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=target_token_count,
        min_length=int(target_token_count * 0.75),
        num_beams=8,
        early_stopping=True,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app layout
st.title("📝 Text Summarizer (64-word BART)")
st.markdown("Enter a passage of text, and the model will generate a ~64-word summary.")

user_input = st.text_area("📄 Enter your text below:", height=300)

if st.button("Generate Summary"):
    if user_input.strip():
        with st.spinner("Generating summary..."):
            result = generate_summary(user_input)
        st.subheader("🧾 Summary:")
        st.success(result)
    else:
        st.warning("Please enter some text first.")
