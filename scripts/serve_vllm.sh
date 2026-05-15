#!/bin/bash
# ─────────────────────────────────────────────
# vLLM 서버 실행 스크립트
# ─────────────────────────────────────────────
# 사용법:
#   bash scripts/serve_vllm.sh                    # 기본값 (Qwen2.5-Coder-7B, port 8000)
#   bash scripts/serve_vllm.sh 8001               # 포트 변경
#   MODEL=deepseek-ai/deepseek-coder-6.7b-instruct bash scripts/serve_vllm.sh  # 모델 변경
#
# 서버 종료: Ctrl+C 또는 kill $(lsof -t -i:8000)
# ─────────────────────────────────────────────
# 서버 띄우기 전에 항상 기존 vLLM 프로세스가 있는지 확인하고 종료
# pkill -f vllm

# 확인용
# lsof -i :8000

# pkill -f python
# pkill -f node

# "microsoft/Phi-3.5-mini-instruct"
# "meta-llama/Llama-3.1-8B-Instruct"
# Qwen/Qwen2.5-Coder-7B-Instruct
# deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
# deepseek-ai/deepseek-coder-7b-instruct-v1.5
MODEL="${MODEL:-deepseek-ai/deepseek-coder-7b-instruct-v1.5}"
PORT="${1:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_UTIL:-0.90}"

echo "============================================"
echo "🚀 vLLM Server"
echo "============================================"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  VRAM  : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Util  : ${GPU_MEMORY_UTILIZATION}"
echo "============================================"
echo ""
echo "서버가 ready 되면 다른 터미널에서 실험을 실행하세요:"
echo "  PYTHONPATH=. python -m src.orchestration.code_then_plan <config.yaml>"
echo ""
echo "health check:"
echo "  curl http://127.0.0.1:${PORT}/v1/models"
echo "============================================"
echo ""

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.80
