import os

import torch
import transformers
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline
from vllm import LLM

os.environ['HF_HOME'] = 'REDACTED'
os.environ['HF_TOKEN'] = 'REDACTED'
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from typing import List

datasets: List[str] = [
    "mmlu",
    "medqa",
    "medbullets",
    "jama"
]

models: List[str] = [
    "azure/o1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/QwQ-32B",
    "Qwen/Qwen2.5-32B-Instruct"
]

tens_parallelism_setting = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 3,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 1,
    "meta-llama/Llama-3.3-70B-Instruct": 3,
    "meta-llama/Llama-3.1-8B-Instruct": 1,
    "Qwen/QwQ-32B": 1,
    "Qwen/Qwen2.5-32B-Instruct": 1
}


class HuggingfaceModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, messages):
        return self.pipeline(messages)


def get_pipeline_deepseek(model_name: str) -> HuggingfaceModelWrapper:
    def set_initialized_submodules(model, state_dict_keys):
        """
        Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
        dict.
        """
        state_dict_keys = set(state_dict_keys)
        not_initialized_submodules = {}
        for module_name, module in model.named_modules():
            if module_name == "":
                # When checking if the root module is loaded there's no need to prepend module_name.
                module_keys = set(module.state_dict())
            else:
                module_keys = {f"{module_name}.{k}" for k in module.state_dict()}
            if module_keys.issubset(state_dict_keys):
                module._is_hf_initialized = True
            else:
                not_initialized_submodules[module_name] = module
        return not_initialized_submodules

    transformers.modeling_utils.set_initialized_submodules = set_initialized_submodules

    ## directly use device_map='auto' if you have enough GPUs
    device_map = {"model.norm": 0, "lm_head": 0, "model.embed_tokens": 0}
    for i in range(61):
        name = "model.layers." + str(i)
        if i < 8:
            device_map[name] = 0
        elif i < 16:
            device_map[name] = 1
        elif i < 25:
            device_map[name] = 2
        elif i < 34:
            device_map[name] = 3
        elif i < 43:
            device_map[name] = 4
        elif i < 52:
            device_map[name] = 5
        elif i < 61:
            device_map[name] = 6

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map
    )

    def forward_hook(module, input, output):
        return torch.clamp(output, -65504, 65504)

    def register_fp16_hooks(model):
        for name, module in model.named_modules():
            if "QuantLinear" in module.__class__.__name__ or isinstance(module, torch.nn.Linear):
                module.register_forward_hook(forward_hook)

    register_fp16_hooks(model)  ##better add this hook to avoid overflow

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return HuggingfaceModelWrapper(model, tokenizer)


def get_pipeline_vllm(model_name: str) -> LLM:
    if not model_name.startswith('opea'):
        # distributed_executor and disable_custom_all_reduce used to support VLLM model unloading see: https://github.com/vllm-project/vllm/issues/1908
        # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        llm = LLM(model=model_name, task="generate", trust_remote_code=True,
                  tensor_parallel_size=tens_parallelism_setting[model_name])  # , distributed_executor_backend="mp", disable_custom_all_reduce=True
    else:
        # llm = LLM(model=model_name, task="generate", trust_remote_code=True, tensor_parallel_size=8, cpu_offload_gb=100) #88432
        llm = LLM(model=model_name, task="generate", trust_remote_code=True, tensor_parallel_size=8,
                  kv_cache_dtype="fp8",
                  calculate_kv_scales=True)
    return llm


def get_pipeline_openai():
    endpoint = os.getenv("ENDPOINT_URL", "REDACTED")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY",
                                 "REDACTED")
    # Initialize Azure OpenAI Service client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-12-01-preview",
    )
    return client
