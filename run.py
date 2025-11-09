import gc
from datetime import datetime as dt
import json
import os
import pickle
import sys
import time
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
from vllm import LLM, SamplingParams

from utils import compute_metrics, image_to_data_url, resize_image
from vmp.utils.energy import EnergyMeter
from vmp.utils.flops import FlopsEstimator

try:
    import pynvml  # type: ignore

    _NVML_OK = True
    try:
        pynvml.nvmlInit()  # type: ignore
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore
    except Exception:
        _NVML_OK = False
        _NVML_HANDLE = None
except Exception:
    _NVML_OK = False
    _NVML_HANDLE = None


def _read_gpu_util_and_mem_pct():
    if not _NVML_OK or _NVML_HANDLE is None:
        return None, None
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)  # type: ignore
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)  # type: ignore
        mem_used_pct = (
            (float(mem_info.used) / float(mem_info.total) * 100.0)
            if getattr(mem_info, "total", 0)
            else 0.0
        )
        return float(util.gpu), float(
            mem_used_pct
        )  # GPU SM utilization %, memory usage %
    except Exception:
        return None, None


load_dotenv()

PATH_DATA = Path("data")
DEBUG = True
AGG_RESULTS_PATH = Path("results") / "aggregated_metrics.csv"
IMAGE_SIZES = [
    (224, 224),
    (336, 336),
    (448, 448),
    (512, 512),
]
PROMPTS = [
    "system_prompt_10",
    "system_prompt_50",
    "system_prompt_100",
    "system_prompt_200",
]
BATCH_SIZES = [
    1,
    2,
    4,
    8,
]
MODELS = [
    "llava-hf/llava-1.5-13b-hf",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/instructblip-vicuna-7b",
    "Salesforce/instructblip-flan-t5-xl",
    "Salesforce/blip2-flan-t5-xl",
    "adept/fuyu-8b",
    "zai-org/cogagent-vqa-hf" "vikhyatk/moondream2",
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

MODELS_RUN = MODELS[:1] if DEBUG else MODELS
BATCH_SIZES_RUN = BATCH_SIZES[-1:] if DEBUG else BATCH_SIZES
IMAGE_SIZES_RUN = IMAGE_SIZES[:1] if DEBUG else IMAGE_SIZES
PROMPTS_RUN = PROMPTS[:1] if DEBUG else PROMPTS
DATASET_DIRS = [d for d in PATH_DATA.iterdir() if d.is_dir()]
if DEBUG:
    DATASET_DIRS = DATASET_DIRS[:1]


def main() -> None:
    for model in MODELS_RUN:
        for batch_size in BATCH_SIZES_RUN:
            llm = LLM(
                model=model,
                # download_dir=os.environ["HF_HOME"],
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
                enable_prefix_caching=True,
                gpu_memory_utilization=0.95,
                max_num_seqs=batch_size,
                # block_size=32,
            )
            try:
                for subdir in DATASET_DIRS:
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
                        for prompt_len in PROMPTS_RUN:
                            system_prompt = dataset_prompt[prompt_len]
                            sampling_params = SamplingParams(
                                temperature=0.0,
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
                            for i in range(0, len(df_samples), batch_size):
                                batch = df_samples.iloc[i : i + batch_size]
                                batch_messages = []
                                for idx, row in batch.iterrows():
                                    image_arr = row["resized_image"]
                                    image_url = image_to_data_url(image_arr)
                                    question_text = row["question"]
                                    choices = row.get("choices", [])
                                    if choices != []:
                                        question_text += (
                                            f"\nChoices: {' '.join(choices)}"
                                        )
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
                                if len(batch_messages) == 0:
                                    continue
                                # measure energy + latency around single chat call
                                _holder = {}

                                def _run_once():
                                    _holder["outputs"] = llm.chat(
                                        batch_messages, sampling_params=sampling_params
                                    )

                                # sample GPU before
                                gpu_before, mem_before = _read_gpu_util_and_mem_pct()
                                t0 = time.perf_counter()
                                ej = energy_meter.integrate_energy_joules(
                                    _run_once, warmup=0, iters=1
                                )
                                t1 = time.perf_counter()
                                # sample GPU after
                                gpu_after, mem_after = _read_gpu_util_and_mem_pct()
                                total_energy_j += float(ej)
                                latency_ms = (t1 - t0) * 1000.0
                                batch_latencies_ms.append(latency_ms)
                                # optional GPU utilization (take max of before/after; mem usage from 'after' if present)
                                sample_gpu_vals = [
                                    v for v in (gpu_before, gpu_after) if v is not None
                                ]
                                if sample_gpu_vals:
                                    gpu_utils.append(max(sample_gpu_vals))
                                if mem_after is not None:
                                    mem_utils.append(mem_after)
                                outputs = _holder["outputs"]
                                for (row_idx, row), output in zip(
                                    batch.iterrows(), outputs
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
                            out_dir = Path("results")
                            out_dir.mkdir(parents=True, exist_ok=True)
                            model_safe = model.replace("/", "_")
                            datetime_str = dt.now().strftime("%y%m%d_%H%M%S")
                            base_name = f"{datetime_str}_{subdir.name}_{model_safe}_{image_size[0]}_{prompt_len}_bs{batch_size}"

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
