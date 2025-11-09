eval "$(conda shell.bash hook)"
conda activate edlm
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_CACHE=/data/ezdoronok/hf_cache
export HF_HOME=/data/ezdoronok/hf_cache
export VLLM_CPU_OMP_THREADS_BIND="48-96"
export VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES=2


vllm serve llava-hf/llava-1.5-13b-hf \
  --host 0.0.0.0 \
  --port 16322 \
  --dtype float16 \
  --gpu_memory_utilization 0.95 \
  --tensor-parallel-size 1 \
  --download-dir /data/ezdoronok/hf_cache \
  --enable-prefix-caching \
  --block-size 32 \
  --enable-log-requests