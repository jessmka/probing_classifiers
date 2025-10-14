# Input prompts to chosen LLM then 
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random

if __name__ == "__main__":
    # Load the model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your desired model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Create a text generation pipeline
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    # Load prompts from a JSON file
    with open('prompt_output.json', 'r') as f:
        prompts_data = json.load(f)

    # Prepare to store results
    results = []

    # Generate responses for each prompt
    for entry in prompts_data:
        for identity, prompts in entry.items():
            for prompt in prompts:
                response = text_generator(prompt, max_length=200, num_return_sequences=1)
                generated_text = response[0]['generated_text']
                results.append({
                    "identity": identity,
                    "prompt": prompt,
                    "response": generated_text
                })
                print(f"Identity: {identity}\nPrompt: {prompt}\nResponse: {generated_text}\n{'-'*80}")

    # Save results to a JSON file
    with open('llm_responses.json', 'w') as f:
        json.dump(results, f, indent=4)