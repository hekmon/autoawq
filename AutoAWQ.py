#!/usr/bin/env python3

import os
import re
import sys

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

## From https://docs.vllm.ai/en/latest/quantization/auto_awq.html
def AutoAWQ(model_path, quant_path):
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Quantize
    model.quantize(tokenizer, quant_config={ "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" })
    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

def main():
    # Handle model name
    if len(sys.argv) - 1 != 1 or not sys.argv[1]:
        print("first argument must be a model path (HF or disk)")
        sys.exit(1)
    model_path = sys.argv[1]
    if os.path.isabs(model_path):
        # assume local file
        if not os.path.isfile(model_path):
            print(f"model '{model_path}' does not exist on disk")
            sys.exit(2)
        # extract model name
        model_name = os.path.basename(model_path)
    else:
        # assume HF
        match = re.search(r'^([a-zA-Z0-9]+)/([a-zA-Z0-9]+)$', model_path)
        if not match:
            print("HF model path must be in the format '<organization>/<repo-name>'")
            sys.exit(2)
        # extract model name
        model_name = match.group(2)
    # Compute output path
    output_dir = os.getenv('AUTOAWQ_OUTPUTDIR')
    if output_dir:
        output_path = f"{output_dir}/{model_name}-awq"
    else:
        output_path = f"{model_name}-awq"
    print(output_path)

if __name__ == "__main__":
    main()
