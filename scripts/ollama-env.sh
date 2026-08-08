#!/usr/bin/env bash
# Thesis-specific Ollama server environment.
# Source this BEFORE starting `ollama serve`:
#
#   source scripts/ollama-env.sh
#   ollama serve
#
# See docs/ollama-config.md for rationale behind each value.
# Do not restart `ollama serve` while Thread A or Thread B has an in-flight run.

# keep models resident for 1h; eliminates the 5-min unload sawtooth.
# Per-request keep_alive from the Python client overrides this; env var is the
# fallback for `ollama run` and external clients.
export OLLAMA_KEEP_ALIVE="1h"

# Allow concurrent inference slots per loaded model. E4B (~4 GB) has plenty of
# headroom for 2; bump 31B cautiously after profiling memory.
export OLLAMA_NUM_PARALLEL=2

# Hold E4B + 31B simultaneously across the Thread A/B crossover window so
# neither has to reload. ~26 GB combined; safe on 64 GB.
export OLLAMA_MAX_LOADED_MODELS=2

# Flash Attention on Apple Metal — compatible with Gemma 4 and with
# logprobs:true + top_logprobs as of Ollama 0.21.x.
export OLLAMA_FLASH_ATTENTION=1

# INTENTIONALLY NOT SET:
# OLLAMA_KV_CACHE_TYPE — quantizing the KV cache (q8_0 / q4_0) perturbs the
# logprob distribution that D6/D9 rest on. Do not enable without running the
# fidelity check in docs/ollama-config.md §6.

echo "Ollama env set — keep_alive=$OLLAMA_KEEP_ALIVE, num_parallel=$OLLAMA_NUM_PARALLEL, max_loaded=$OLLAMA_MAX_LOADED_MODELS, flash_attn=$OLLAMA_FLASH_ATTENTION"
