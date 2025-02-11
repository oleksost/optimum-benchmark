#!/bin/bash

# Define specific combinations of parameters
# combinations=(
#     "scenario.generate_kwargs.max_new_tokens=10 scenario.generate_kwargs.min_new_tokens=10"
#     "scenario.generate_kwargs.max_new_tokens=100 scenario.generate_kwargs.min_new_tokens=100"
#     "scenario.generate_kwargs.max_new_tokens=500 scenario.generate_kwargs.min_new_tokens=500"
#     "scenario.generate_kwargs.max_new_tokens=1000 scenario.generate_kwargs.min_new_tokens=1000"
# )

# # Define benchmark configurations
new_tokens=(10 100 200 400 1000)

configs=(
    # "cuda_pytorch_hymba"
    # "cuda_pytorch_llama_32_1B"
    "cuda_pytorch_llama_32_3B"
    "cuda_pytorch_pythia_2b"
    "cuda_pytorch_mamba2.8-hf"
    "cuda_pytorch_mamba2-2.7b-hf"
    "cuda_pytorch_mamba2.8-mamba_repo"
    "cuda_pytorch_mamba2-2.7-mamba_repo"
)

# Iterate over each combination and configuration
for nt in "${new_tokens[@]}"; do
    for config in "${configs[@]}"; do
        run_name="${config}_ntok${nt}_bs32"
        echo "Running: optimum-benchmark --config-dir benchmarks --config-name $config name=$run_name +new_tokens=$nt scenario.input_shapes.batch_size=32"
        optimum-benchmark --config-dir benchmarks/ --config-name "$config" name="$run_name" +new_tokens="$nt" scenario.input_shapes.batch_size=32
        wait
    done
done

# nohup optimum-benchmark --config-dir benchmarks/ --config-name cuda_pytorch_inference -m backend.model="meta-llama/Llama-3.2-1B" scenario.input_shapes.batch_size=1,32 +new_tokens=10,50,100,200 >benchmark.log 2>&1 &
# optimum-benchmark --config-dir benchmarks/ --config-name cuda_pytorch_inference -m backend.model="nvidia/Hymba-1.5B-Base","meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B" scenario.input_shapes.batch_size=8,32 +new_tokens=10,100,500,1000
# wait
# optimum-benchmark --config-dir benchmarks/ --config-name cuda_pytorch_llama_32_1B -m scenario.input_shapes.batch_size=8,32 +new_tokens=10,100,500,1000
# wait
# optimum-benchmark --config-dir benchmarks/ --config-name cuda_pytorch_llama_32_3B -m scenario.input_shapes.batch_size=8,32 +new_tokens=10,100,500,1000