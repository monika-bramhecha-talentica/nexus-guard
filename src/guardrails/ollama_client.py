"""
Phase 5: Ollama LLM Client Integration

Provides async LLM evaluation calls to Ollama with fallback to rule-based scoring.
Uses Mistral 7B model for efficient, fast response evaluation.

Features:
  - Async evaluation calls to Ollama
  - Graceful fallback to rule-based scoring if Ollama unavailable
  - Response parsing and validation (JSON format)
  - Error handling and logging
  - Connection timeout and retry logic
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Ollama client configuration."""

    host: str = "localhost"
    port: int = 11434
    model_name: str = "mistral"
    timeout: int = 10  # seconds
    max_retries: int = 2
    fallback_enabled: bool = True

    @property
    def base_url(self) -> str:
        """Construct base URL from host and port."""
        return f"http://{self.host}:{self.port}"


class OllamaClient:
    """
    Async client for Ollama LLM evaluation.

    Handles communication with Ollama server, manages timeouts and retries,
    and provides graceful fallback to rule-based scoring when LLM unavailable.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.

        Args:
            config: OllamaConfig with connection parameters
        """
        self.config = config or OllamaConfig()
        self.is_available = False
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"OllamaClient initialized: {self.config.base_url}/api/generate "
            f"model={self.config.model_name}"
        )

    async def check_connectivity(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if Ollama is reachable, False otherwise
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    self.is_available = response.status == 200
                    logger.info(f"Ollama connectivity check: {self.is_available}")
                    return self.is_available
        except Exception as e:
            logger.warning(f"Ollama connectivity check failed: {e}")
            self.is_available = False
            return False

    async def evaluate_llm(
        self,
        response: str,
        query: str = "",
        agent_id: str = "agent_unknown",
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Evaluate response using Ollama LLM.

        Args:
            response: Response text to evaluate
            query: Original query (for context)
            agent_id: ID of evaluating agent

        Returns:
            Tuple of (evaluation_dict, success_flag)
            evaluation_dict contains: accuracy, relevance, completeness, safety,
            hallucination, pii_handling scores (0.0-1.0 each)
        """
        if not self.is_available:
            logger.warning("Ollama not available for evaluation")
            return {}, False

        # Check cache first
        response_hash = hash(response)
        if response_hash in self.evaluation_cache:
            logger.debug(f"Cache hit for evaluation (hash={response_hash})")
            return self.evaluation_cache[response_hash], True

        prompt = self._build_evaluation_prompt(response, query)

        try:
            evaluation_dict = await self._call_ollama(prompt)
            if evaluation_dict:
                # Cache the result
                self.evaluation_cache[response_hash] = evaluation_dict
                logger.info(f"LLM evaluation successful: {agent_id}")
                return evaluation_dict, True
            else:
                logger.warning("LLM evaluation returned empty result")
                return {}, False
        except asyncio.TimeoutError:
            logger.warning(f"Ollama evaluation timeout after {self.config.timeout}s")
            return {}, False
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return {}, False

    def _build_evaluation_prompt(self, response: str, query: str = "") -> str:
        """
        Build optimized evaluation prompt for Ollama.

        Prompt is designed to be token-efficient (<500 tokens) while maintaining
        clear evaluation criteria.

        Args:
            response: Response to evaluate
            query: Original query for context

        Returns:
            Structured evaluation prompt
        """
        return f"""Evaluate this response on six dimensions (0.0-1.0 scale).

Query: {query if query else "No specific query"}

Response: {response}

Provide JSON evaluation:
{{
  "accuracy_score": <0.0-1.0>,
  "relevance_score": <0.0-1.0>,
  "completeness_score": <0.0-1.0>,
  "safety_score": <0.0-1.0>,
  "hallucination_score": <0.0-1.0>,
  "pii_handling_score": <0.0-1.0>,
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>"],
  "hallucinations_detected": ["<hallucination1>" or "none"]
}}"""

    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """
        Make async call to Ollama API.

        Args:
            prompt: Evaluation prompt to send

        Returns:
            Parsed evaluation dictionary or empty dict on failure
        """
        import aiohttp

        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_llm_response(data.get("response", ""))
                        else:
                            logger.warning(
                                f"Ollama returned status {response.status} (attempt {attempt + 1})"
                            )
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except Exception as e:
                logger.warning(f"Ollama call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5)

        return {}

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into evaluation dictionary.

        Handles various response formats and validates JSON structure.

        Args:
            response_text: Raw response from LLM

        Returns:
            Validated evaluation dictionary with required fields
        """
        try:
            # Try to extract JSON from response
            # LLM might include explanatory text before/after JSON
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return {}

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Validate required fields
            required_fields = [
                "accuracy_score",
                "relevance_score",
                "completeness_score",
                "safety_score",
                "hallucination_score",
                "pii_handling_score",
            ]

            # Normalize field names and values
            normalized = {}
            for field in required_fields:
                if field in data:
                    value = float(data[field])
                    normalized[field] = max(0.0, min(1.0, value))  # Clamp to [0, 1]
                else:
                    logger.warning(f"Missing required field: {field}")
                    return {}  # Fail if required field missing

            # Add optional fields
            normalized["strengths"] = data.get("strengths", [])
            normalized["weaknesses"] = data.get("weaknesses", [])
            normalized["hallucinations_detected"] = data.get(
                "hallucinations_detected", []
            )

            logger.info(f"LLM response parsed successfully: accuracy={normalized['accuracy_score']:.2f}")
            return normalized

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return {}
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid score values in LLM response: {e}")
            return {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get evaluation cache statistics.

        Returns:
            Dictionary with cache stats (size, entries)
        """
        return {
            "cache_entries": len(self.evaluation_cache),
            "cache_size_bytes": sum(
                len(json.dumps(v)) for v in self.evaluation_cache.values()
            ),
            "ollama_available": self.is_available,
        }

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        logger.info("Evaluation cache cleared")
