import gc
import logging
import json
import os
import pickle
import sys
import threading
import time
import typing as tp
from datetime import datetime as dt
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import compute_metrics, image_to_data_url, resize_image
from vmp.utils.energy import EnergyMeter
from vmp.utils.flops import FlopsEstimator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NVML_HANDLES: tp.List[tp.Any] = []

try:
    import pynvml  # type: ignore

    _NVML_OK = True
    try:
        pynvml.nvmlInit()  # type: ignore
        _visible_devices_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if _visible_devices_env:
            _visible_indices = []
            for _part in _visible_devices_env.split(","):
                _part = _part.strip()
                if not _part or _part.lower() == "none":
                    continue
                try:
                    _visible_indices.append(int(_part))
                except ValueError:
                    continue
        else:
            _visible_indices = list(range(pynvml.nvmlDeviceGetCount()))  # type: ignore
        _NVML_HANDLES = []
        for _idx in _visible_indices:
            try:
                _NVML_HANDLES.append(
                    pynvml.nvmlDeviceGetHandleByIndex(_idx)  # type: ignore
                )
            except Exception:
                continue
        if not _NVML_HANDLES:
            _NVML_OK = False
    except Exception:
        _NVML_OK = False
        _NVML_HANDLES = []
except Exception:
    _NVML_OK = False
    _NVML_HANDLES = []


def _read_gpu_util_and_mem_pct():
    if not _NVML_OK or not _NVML_HANDLES:
        return None, None
    gpu_vals: tp.List[float] = []
    mem_vals: tp.List[float] = []
    for handle in _NVML_HANDLES:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore
        except Exception:
            continue
        mem_total = float(getattr(mem_info, "total", 0.0))
        mem_used_pct = (float(mem_info.used) / mem_total * 100.0) if mem_total else 0.0
        gpu_vals.append(float(util.gpu))
        mem_vals.append(float(mem_used_pct))
    if not gpu_vals and not mem_vals:
        return None, None
    gpu_val = max(gpu_vals) if gpu_vals else None
    mem_val = max(mem_vals) if mem_vals else None
    return gpu_val, mem_val


def _measure_energy_and_gpu(
    energy_meter: EnergyMeter,
    fn: tp.Callable,
    *,
    warmup: int = 0,
    iters: int = 1,
    poll_interval: float = 0.1,
):
    gpu_samples = []
    mem_samples = []

    sampling_enabled = _NVML_OK and bool(_NVML_HANDLES)
    if not sampling_enabled:
        energy_j = energy_meter.integrate_energy_joules(fn, warmup=warmup, iters=iters)
        return energy_j, gpu_samples, mem_samples

    stop_event = threading.Event()
    post_sample_sleep = min(0.5, max(0.05, poll_interval * 2.0))

    def _sample_once():
        gpu, mem = _read_gpu_util_and_mem_pct()
        if gpu is not None:
            gpu_samples.append(gpu)
        if mem is not None:
            mem_samples.append(mem)

    def _sampler():
        while True:
            _sample_once()
            if stop_event.wait(poll_interval):
                _sample_once()
                break

    sampler_thread = threading.Thread(
        target=_sampler, name="gpu-metrics-sampler", daemon=True
    )
    sampler_thread.start()
    try:
        energy_j = energy_meter.integrate_energy_joules(fn, warmup=warmup, iters=iters)
    finally:
        if post_sample_sleep > 0:
            time.sleep(post_sample_sleep)
        stop_event.set()
        sampler_thread.join()
    return energy_j, gpu_samples, mem_samples


load_dotenv()

# Set up paths
PATH_ROOT = Path(__file__).parent
PATH_DATA = PATH_ROOT / "data"
PATH_RESULTS = PATH_ROOT / "results"
PATH_PROFILING = PATH_RESULTS / "profiling"
for path in [PATH_DATA, PATH_RESULTS, PATH_PROFILING]:
    path.mkdir(parents=True, exist_ok=True)

DEBUG = False
GPU_SAMPLING_INTERVAL_S = float(os.environ.get("GPU_SAMPLING_INTERVAL_S", "0.01"))
AGG_RESULTS_PATH = PATH_RESULTS / "aggregated_metrics.csv"

IMAGE_SIZES = [
    (224, 224),
    (336, 336),
    (448, 448),
    (512, 512),
]

PROMPTS = [
    # "system_prompt_10",
    # "system_prompt_50",
    "system_prompt_100",
    # "system_prompt_200",
]

BATCH_SIZES = [
    # 1,
    # 4,
    # 16,
    64,
    # 320,
]

MODELS = [
    "llava-hf/llava-1.5-13b-hf",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/instructblip-vicuna-7b",
    "Salesforce/instructblip-flan-t5-xl",
    "Salesforce/blip2-flan-t5-xl",
    "adept/fuyu-8b",
    "zai-org/cogagent-vqa-hf",
    "vikhyatk/moondream2",
    "HuggingFaceM4/idefics2-8b",
]

PARAMS_B = {
    "llava-hf/llava-1.5-13b-hf": 13.0,
    "Salesforce/blip2-opt-2.7b": 2.7,
    "Salesforce/instructblip-vicuna-7b": 7.0,
    "Salesforce/instructblip-flan-t5-xl": 3.0,
    "Salesforce/blip2-flan-t5-xl": 3.0,
    "adept/fuyu-8b": 8.0,
    "zai-org/cogagent-vqa-hf": 18.0,  # rough
    "vikhyatk/moondream2": 2.0,  # rough
    "HuggingFaceM4/idefics2-8b": 8.0,
}

ANSWERS_COL_MAP = {
    "coco": "answer",
    "scienceqa": "answer",
    "textvqa": "answers",
}

RUN_TIMESTAMP = dt.now().strftime("%y%m%d_%H%M%S")

VLLM_ENABLE_PROFILING = int(os.environ.get("VLLM_ENABLE_PROFILING", 0))

if VLLM_ENABLE_PROFILING:
    print("VLLM profiling is enabled")
    run_timestamp_dir = PATH_PROFILING / RUN_TIMESTAMP
    run_timestamp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = run_timestamp_dir.as_posix()
else:
    print("VLLM profiling is disabled")

MODELS_RUN = MODELS[:1] if DEBUG else MODELS
BATCH_SIZES_RUN = BATCH_SIZES[-1:] if DEBUG else BATCH_SIZES
IMAGE_SIZES_RUN = IMAGE_SIZES[:1] if DEBUG else IMAGE_SIZES
PROMPTS_RUN = PROMPTS[:1] if DEBUG else PROMPTS
DATASET_DIRS = [d for d in PATH_DATA.iterdir() if d.is_dir()]
if DEBUG:
    DATASET_DIRS = DATASET_DIRS[:1]


def main() -> None:
    for model in tqdm(MODELS_RUN, desc="Processing model", total=len(MODELS_RUN)):
        for batch_size in tqdm(
            BATCH_SIZES_RUN, desc="Processing batch size", total=len(BATCH_SIZES_RUN)
        ):
            llm = LLM(
                model=model,
                # download_dir=os.environ["HF_HOME"],
                trust_remote_code=True,
                # tensor_parallel_size=torch.cuda.device_count(),
                enable_prefix_caching=True,
                gpu_memory_utilization=0.95,
                max_num_seqs=batch_size,
                # block_size=32,
            )
            try:
                for subdir in tqdm(
                    DATASET_DIRS, desc="Processing dataset", total=len(DATASET_DIRS)
                ):
                    with open(subdir / "samples_300.pkl", "rb") as f:
                        samples = pickle.load(f)
                    df_samples = pd.DataFrame(samples)
                    if DEBUG:
                        df_samples = df_samples.sample(frac=0.2, random_state=42)
                    with open(f"prompts/{subdir.name}.yaml", "r") as f:
                        dataset_prompt = yaml.safe_load(f)
                    for image_size in IMAGE_SIZES_RUN:
                        resized_images = df_samples["image"].apply(
                            lambda x: resize_image(np.array(x), image_size)
                        )
                        df_samples = df_samples.copy()
                        df_samples["resized_image"] = resized_images
                        for prompt_len in tqdm(
                            PROMPTS_RUN,
                            desc="Processing prompt",
                            total=len(PROMPTS_RUN),
                        ):
                            system_prompt = dataset_prompt[prompt_len]
                            sampling_params = SamplingParams(
                                temperature=0.01,
                                top_p=1.0,
                                max_tokens=256,
                            )
                            results = []
                            # perf accumulators
                            batch_latencies_ms = []
                            total_energy_j = 0.0
                            gpu_utils = []
                            mem_utils = []
                            total_gen_tokens = 0
                            total_est_tflops = 0.0
                            energy_meter = EnergyMeter()
                            flops_estimator = FlopsEstimator()
                            for i in tqdm(
                                range(0, len(df_samples), batch_size),
                                desc="Processing batch",
                                total=len(df_samples) // batch_size,
                            ):
                                batch = df_samples.iloc[i : i + batch_size]
                                batch_messages = []
                                batch_entries = []
                                for idx, row in batch.iterrows():
                                    image_arr = row["resized_image"]
                                    image_url = image_to_data_url(image_arr)
                                    question_text = row["question"]
                                    choices = row.get("choices", [])
                                    if choices != []:
                                        choices_text = "\n".join(
                                            f"{i+1}. {c}" for i, c in enumerate(choices)
                                        )
                                        question_text += f"\n\nChoices:\n{choices_text}"
                                    question_text = question_text.strip() + "\n"
                                    user_content = []
                                    if len(question_text) > 0:
                                        user_content.append(
                                            {"type": "text", "text": question_text}
                                        )
                                    user_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": image_url},
                                        }
                                    )
                                    batch_messages.append(
                                        [
                                            {
                                                "role": "system",
                                                "content": system_prompt,
                                            },
                                            {"role": "user", "content": user_content},
                                        ]
                                    )
                                    batch_entries.append((idx, row, question_text))
                                if len(batch_entries) == 0:
                                    continue
                                # measure energy + latency around single chat call
                                _holder = {}

                                def _run_once():
                                    if VLLM_ENABLE_PROFILING:
                                        llm.start_profile()
                                    try:
                                        outputs = llm.chat(
                                            batch_messages,
                                            sampling_params=sampling_params,
                                        )
                                        _holder["outputs"] = outputs
                                        # print(f"Outputs: {outputs}")

                                    except Exception as e:
                                        print(f"Error during profiling: {e}")
                                    finally:
                                        if VLLM_ENABLE_PROFILING:
                                            llm.stop_profile()

                                t0 = time.perf_counter()
                                ej, gpu_samples_batch, mem_samples_batch = (
                                    _measure_energy_and_gpu(
                                        energy_meter,
                                        _run_once,
                                        warmup=0,
                                        iters=1,
                                        poll_interval=GPU_SAMPLING_INTERVAL_S,
                                    )
                                )
                                t1 = time.perf_counter()
                                total_energy_j += float(ej)
                                latency_ms = (t1 - t0) * 1000.0
                                batch_latencies_ms.append(latency_ms)
                                if gpu_samples_batch:
                                    gpu_utils.extend(gpu_samples_batch)
                                if mem_samples_batch:
                                    mem_utils.extend(mem_samples_batch)
                                outputs = _holder["outputs"]
                                for (row_idx, row, question_text), output in zip(
                                    batch_entries, outputs
                                ):
                                    text = (
                                        output.outputs[0].text
                                        if hasattr(output, "outputs")
                                        and len(output.outputs) > 0
                                        else ""
                                    )
                                    results.append(
                                        {
                                            "index": int(row_idx),
                                            "dataset": subdir.name,
                                            "model": model,
                                            "batch_size": batch_size,
                                            "image_width": int(image_size[0]),
                                            "image_height": int(image_size[1]),
                                            "prompt_len": prompt_len,
                                            "question": question_text,
                                            "answer": row.get("answer", "")
                                            or row.get("answers", []),
                                            "response": text,
                                        }
                                    )
                                # token and FLOPs estimation per batch
                                try:
                                    gen_tokens_batch = sum(
                                        (
                                            len((o.outputs[0].text or "").split())
                                            if hasattr(o, "outputs")
                                            and len(o.outputs) > 0
                                            else 0
                                        )
                                        for o in outputs
                                    )
                                except Exception:
                                    gen_tokens_batch = (
                                        sum(
                                            len((r.get("response", "") or "").split())
                                            for r in results[-len(outputs) :]
                                        )
                                        if len(results) >= len(outputs)
                                        else 0
                                    )
                                total_gen_tokens += int(gen_tokens_batch)
                                image_res = int(max(image_size[0], image_size[1]))
                                model_params_b = float(PARAMS_B.get(model, 0.0))
                                est = flops_estimator.estimate(
                                    num_tokens=int(gen_tokens_batch),
                                    image_resolution=image_res,
                                    model_param_count_b=model_params_b,
                                )
                                total_est_tflops += float(est.total_tflops)

                            out_dir = PATH_RESULTS
                            out_dir.mkdir(parents=True, exist_ok=True)
                            model_safe = model.replace("/", "_")
                            base_name = f"{RUN_TIMESTAMP}_{subdir.name}_{model_safe}_{image_size[0]}_{prompt_len}_bs{batch_size}"

                            # raw results only in debug mode
                            if DEBUG:
                                raw_path = out_dir / f"{base_name}_raw.csv"
                                pd.DataFrame(results).to_csv(raw_path, index=False)

                            # compute and save aggregated metric
                            metric_value = float("nan")
                            df_res = pd.DataFrame(results)
                            metric_value = compute_metrics(
                                subdir.name, df_res, "response", "answer"
                            )

                            df_metric_row = pd.DataFrame(
                                [
                                    {
                                        "debug": DEBUG,
                                        "dataset": subdir.name,
                                        "model": model,
                                        "batch_size": batch_size,
                                        "image_width": int(image_size[0]),
                                        "image_height": int(image_size[1]),
                                        "prompt_len": prompt_len,
                                        "metric": metric_value,
                                        "latency_ms_mean": (
                                            (
                                                sum(batch_latencies_ms)
                                                / len(batch_latencies_ms)
                                            )
                                            if batch_latencies_ms
                                            else float("nan")
                                        ),
                                        "gpu_util_mean": (
                                            (sum(gpu_utils) / len(gpu_utils))
                                            if gpu_utils
                                            else float("nan")
                                        ),
                                        "mem_util_mean": (
                                            (sum(mem_utils) / len(mem_utils))
                                            if mem_utils
                                            else float("nan")
                                        ),
                                        "energy_j": total_energy_j,
                                        "est_tflops_total": total_est_tflops,
                                        "gen_tokens_total": total_gen_tokens,
                                    }
                                ]
                            )
                            header_needed = not AGG_RESULTS_PATH.exists()
                            df_metric_row.to_csv(
                                AGG_RESULTS_PATH,
                                mode="a",
                                header=header_needed,
                                index=False,
                            )
                            print(
                                f"[AGG] dataset={subdir.name} model={model_safe} size={image_size[0]}x{image_size[1]} prompt={prompt_len} bs={batch_size} metric={metric_value}"
                            )
            finally:
                del llm
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        freeze_support()
    except Exception:
        pass
    main()
