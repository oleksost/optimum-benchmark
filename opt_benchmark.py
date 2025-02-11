from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    TorchrunConfig,
    InferenceConfig,
    PyTorchConfig,
)
from optimum_benchmark.logging_utils import setup_logging
import torch


setup_logging(level="INFO")

if __name__ == "__main__":
    model_path = "/home/toolkit/dev/optimum-benchmark/models/Hymba-1.5B-Base/"
    launcher_config = TorchrunConfig(nproc_per_node=1)
    scenario_config = InferenceConfig(
        latency=True,
        memory=True,
        input_shapes={"sequence_length": 200, "batch_size": 2},
        generate_kwargs={
            "top_p": 0.9,
            "top_k": 1,
            "cg": True,
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 200,
            "min_new_tokens": 200,
        },
    )
    # backend_config = PyTorchConfig(model="gpt2", device="cuda", device_ids="0", no_weights=True)
    backend_config = PyTorchConfig(
        model="state-spaces/mamba-2.8b",  # "nvidia/Hymba-1.5B-Base",
        library="transformers",
        device="cuda",
        # torch_dtype="bfloat16",
        model_type="mamba_ssm_repo",
        device_ids="0",
        no_weights=False,
        model_kwargs={
            "dtype": "bfloat16",
            # "trust_remote_code": True,
            # "use_cache": True,
        },
    )

    # backend_config = PyTorchConfig(model="meta-llama/Llama-3.2-3B",
    #                                library="transformers", device="cuda", torch_dtype="bfloat16",
    #                                device_ids="0", no_weights=True, model_kwargs={"trust_remote_code": True, "use_cache": True})
    benchmark_config = BenchmarkConfig(
        name="pytorch_gpt2",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)

    # convert artifacts to a dictionary or dataframe
    benchmark_config.to_dict()  # or benchmark_config.to_dataframe()

    # save artifacts to disk as json or csv files
    benchmark_report.save_csv(
        "benchmark_report.csv"
    )  # or benchmark_report.save_json("benchmark_report.json")

    # push artifacts to the hub
    # benchmark_config.push_to_hub("IlyasMoutawwakil/pytorch_gpt2") # or benchmark_config.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")

    # or merge them into a single artifact
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.save_json("benchmark.json")  # or benchmark.save_csv("benchmark.csv")
    # benchmark.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")

    # load artifacts from the hub
    # benchmark = Benchmark.from_hub("IlyasMoutawwakil/pytorch_gpt2") # or Benchmark.from_hub("IlyasMoutawwakil/pytorch_gpt2")

    # # or load them from disk
    # benchmark = Benchmark.load_json("benchmark.json") # or Benchmark.load_csv("benchmark_report.csv")
