"""MCP-specific configuration from config.yaml."""

from ..core.config import RAW as _raw

EMBEDDING_DIM = 1024
DEFAULT_SEARCH_LIMIT = _raw.get("embedding_search_limit", 10)
SEARCH_SIMILARITY_FLOOR = _raw.get("search_similarity_floor", 0.4)
SNIPPET_MAX_CHARS = _raw.get("snippet_max_chars_mcp", 500)

# Clustering
PCA_N_COMPONENTS = _raw.get("pca_n_components", 15)
TIME_WEIGHT = _raw.get("time_weight", 2.0)
CLUSTER_MIN_SIZE = _raw.get("cluster_min_size", 3)
CLUSTER_MIN_SAMPLES = _raw.get("cluster_min_samples", None)  # defaults to CLUSTER_MIN_SIZE
TOPIC_SIM_THRESHOLD_PCT = _raw.get("topic_sim_threshold_pct", 0.30)
COARSE_SIMILARITY_FLOOR = _raw.get("coarse_similarity_floor", 0.15)  # SQL pre-filter: deliberately loose
SUMMARIZE_SNIPPETS = _raw.get("summarize_snippets", 0)

# Vision-language model (tsa-describe)
VLM_MODEL_ID = _raw.get(
    "vlm_model_id", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
)
VLM_MAX_TOKENS = _raw.get("vlm_max_tokens", 500)
VLM_REPETITION_PENALTY = _raw.get("vlm_repetition_penalty", 1.3)
