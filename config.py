"""
Nexus Guard Configuration

Central configuration for all system components.
All values can be overridden via environment variables.
"""

import os
from typing import Dict, Any

# ==============================================================================
# Environment & Deployment
# ==============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ==============================================================================
# Server & API
# ==============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_RELOAD = DEBUG

WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "600"))  # 10 minutes

# ==============================================================================
# Ollama & Judge LLM
# ==============================================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
JUDGE_TIMEOUT_SEC = int(os.getenv("JUDGE_TIMEOUT_SEC", "5"))
JUDGE_BATCH_SIZE = int(os.getenv("JUDGE_BATCH_SIZE", "1"))

# ==============================================================================
# Orchestrator Settings
# ==============================================================================

DEFAULT_TOKEN_BUDGET = int(os.getenv("DEFAULT_TOKEN_BUDGET", "5000"))
MAX_AGENT_HOPS = int(os.getenv("MAX_AGENT_HOPS", "3"))
TOKEN_BUDGET_WARNING_PERCENT = int(os.getenv("TOKEN_BUDGET_WARNING_PERCENT", "80"))

# ==============================================================================
# PII Masking Settings
# ==============================================================================

PII_MASKING_ENABLED = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
PII_DETECTION_THRESHOLD = float(os.getenv("PII_DETECTION_THRESHOLD", "0.95"))
PII_MASKING_BUFFER_SIZE = int(os.getenv("PII_MASKING_BUFFER_SIZE", "25"))  # tokens

# Custom PII patterns (JSON string)
CUSTOM_PII_PATTERNS = os.getenv(
    "CUSTOM_PII_PATTERNS",
    '{"CUSTOM_ID": "XXXX\\\\d{8}"}'  # Example: custom ID format
)

# ==============================================================================
# Redis Configuration (Session/Cache)
# ==============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))  # 1 hour

# ==============================================================================
# Audit & Logging
# ==============================================================================

AUDIT_LOG_ENABLED = os.getenv("AUDIT_LOG_ENABLED", "true").lower() == "true"
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "logs/audit.jsonl")
STRUCTURED_LOGGING = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"

# ==============================================================================
# Monitoring & Metrics
# ==============================================================================

PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
METRICS_RETENTION_HOURS = int(os.getenv("METRICS_RETENTION_HOURS", "24"))

# ==============================================================================
# Feature Flags
# ==============================================================================

FEATURE_AUTO_CORRECTION = os.getenv("FEATURE_AUTO_CORRECTION", "true").lower() == "true"
FEATURE_STREAMING = os.getenv("FEATURE_STREAMING", "true").lower() == "true"
FEATURE_CACHING = os.getenv("FEATURE_CACHING", "true").lower() == "true"

# ==============================================================================
# Performance Targets
# ==============================================================================

LATENCY_P95_TARGET_MS = int(os.getenv("LATENCY_P95_TARGET_MS", "200"))
PII_MASKING_LATENCY_TARGET_MS = int(os.getenv("PII_MASKING_LATENCY_TARGET_MS", "50"))
JUDGE_LATENCY_TARGET_MS = int(os.getenv("JUDGE_LATENCY_TARGET_MS", "500"))

# ==============================================================================
# Build Configuration Dictionary
# ==============================================================================

CONFIG: Dict[str, Any] = {
    # Environment
    "environment": ENVIRONMENT,
    "debug": DEBUG,
    "log_level": LOG_LEVEL,

    # Server
    "api_host": API_HOST,
    "api_port": API_PORT,
    "api_workers": API_WORKERS,
    "websocket_timeout": WEBSOCKET_TIMEOUT,

    # Ollama
    "ollama_host": OLLAMA_HOST,
    "ollama_model": OLLAMA_MODEL,
    "judge_timeout_sec": JUDGE_TIMEOUT_SEC,

    # Orchestrator
    "default_token_budget": DEFAULT_TOKEN_BUDGET,
    "max_agent_hops": MAX_AGENT_HOPS,
    "token_budget_warning_percent": TOKEN_BUDGET_WARNING_PERCENT,

    # PII
    "pii_masking_enabled": PII_MASKING_ENABLED,
    "pii_detection_threshold": PII_DETECTION_THRESHOLD,
    "pii_masking_buffer_size": PII_MASKING_BUFFER_SIZE,

    # Redis
    "redis_host": REDIS_HOST,
    "redis_port": REDIS_PORT,
    "redis_db": REDIS_DB,
    "redis_password": REDIS_PASSWORD,
    "redis_ttl_seconds": REDIS_TTL_SECONDS,

    # Audit
    "audit_log_enabled": AUDIT_LOG_ENABLED,
    "audit_log_file": AUDIT_LOG_FILE,
    "structured_logging": STRUCTURED_LOGGING,

    # Monitoring
    "prometheus_enabled": PROMETHEUS_ENABLED,
    "prometheus_port": PROMETHEUS_PORT,

    # Features
    "feature_auto_correction": FEATURE_AUTO_CORRECTION,
    "feature_streaming": FEATURE_STREAMING,
    "feature_caching": FEATURE_CACHING,

    # Performance
    "latency_p95_target_ms": LATENCY_P95_TARGET_MS,
    "pii_masking_latency_target_ms": PII_MASKING_LATENCY_TARGET_MS,
    "judge_latency_target_ms": JUDGE_LATENCY_TARGET_MS,
}


def get_config() -> Dict[str, Any]:
    """Get full configuration dictionary."""
    return CONFIG


def get_redis_url() -> str:
    """Build Redis connection URL."""
    password = f":{REDIS_PASSWORD}@" if REDIS_PASSWORD else ""
    return f"redis://{password}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"


def get_ollama_url() -> str:
    """Get Ollama server URL."""
    return OLLAMA_HOST


if __name__ == "__main__":
    # Print configuration on startup
    import json
    print("=" * 80)
    print("NEXUS GUARD CONFIGURATION")
    print("=" * 80)
    print(json.dumps(CONFIG, indent=2, default=str))
    print("=" * 80)
