import torch
from airllm import AutoModel

class QwenAirLLMWrapper:
    def __init__(self, model_id="Qwen/Qwen2.5-32B-Instruct", compression="4bit"):
        print(f"Initializing AirLLM for {model_id} with {compression} compression...")
        self.model = AutoModel.from_pretrained(
            model_id,
            compression=compression,
            delete_original=True,
            layer_shards_saving_path="./model_shards"
        )
        print("Model initialized successfully.")

    def generate(self, prompt, max_new_tokens=256, temperature=0.7):
        input_tokens = self.model.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=4096,
            padding=False
        )

        generation_output = self.model.generate(
            input_tokens['input_ids'].cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True
        )

        output_text = self.model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        return output_text

if __name__ == "__main__":
    # Test initialization (this will trigger the download and sharding)
    qwen = QwenAirLLMWrapper()
    test_prompt = "Merhaba, nasılsın? Adaptive RAG sistemi hakkında ne biliyorsun?"
    print(f"Prompt: {test_prompt}")
    response = qwen.generate(test_prompt)
    print(f"Response: {response}")
