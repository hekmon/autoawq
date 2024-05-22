#!/usr/bin/env python3

import os
import re
import sys
import time

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
        match = re.search(r'^([a-zA-Z0-9\-]+)/([a-zA-Z0-9\-]+)$', model_path)
        if not match:
            print("HF model path must be in the format '<organization>/<repo-name>'")
            sys.exit(2)
        # extract model name
        model_name = match.group(2)
    # Compute output path
    output_path = f"{model_name}-awq"
    output_dir = os.getenv('AUTOAWQ_OUTPUTDIR')
    if output_dir:
        # Check if directory exists
        if not os.path.isdir(output_dir):
            print(f"AUTOAWQ_OUTPUTDIR is set to '{output_dir}' but directory does not exist.")
            sys.exit(3)
        # Append model name to path
        output_path = os.path.join(output_dir, output_path)
    # Print params handling results
    print(f"The model '{model_path}' will be quantized with AWQ in '{output_path}'")
    print()
    # Call the function
    start_time = time.time()
    AutoAWQ(model_path, output_path)
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print()
    print(f"The AWQ quantization took {int(elapsed_seconds // 60)} minute(s) and {int(elapsed_seconds % 60)} second(s)")

if __name__ == "__main__":
    main()
