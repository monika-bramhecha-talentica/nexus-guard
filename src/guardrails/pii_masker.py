"""
Phase 3.2: PII Masking Engine

Anonymizes detected PII in real-time with 4 masking strategies:
1. Placeholder: [MASKED_TYPE]
2. Partial: Show first/last 4 characters
3. Hash: SHA256 with prefix
4. Replacement: Consistent token mapping
"""

import hashlib
import logging
from typing import List, Dict, Optional
from .pii_detector import PIIEntity

logger = logging.getLogger(__name__)


class PIIMasker:
    """
    Masking engine for anonymizing PII entities.

    Supports 4 strategies:
    - placeholder: Replace with [MASKED_TYPE]
    - partial: Keep first 4, last 4 characters
    - hash: SHA256 hash with prefix
    - replacement: Consistent token mapping
    """

    # Strategy options
    VALID_STRATEGIES = ["placeholder", "partial", "hash", "replacement"]

    def __init__(
        self,
        strategy: str = "placeholder",
        confidence_threshold: float = 0.75
    ):
        """
        Initialize masker with chosen strategy.

        Args:
            strategy: Masking strategy ('placeholder', 'partial', 'hash', 'replacement')
            confidence_threshold: Minimum confidence to mask (0.0-1.0)

        Raises:
            ValueError: If strategy not in VALID_STRATEGIES
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Strategy '{strategy}' not in {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.replacement_map: Dict[str, str] = {}  # For 'replacement' strategy
        self.token_counter = 0

        logger.info(
            f"PIIMasker initialized: strategy={strategy}, "
            f"threshold={confidence_threshold}"
        )

    def mask_text(self, text: str, entities: List[PIIEntity]) -> str:
        """
        Mask detected PII entities in text.

        Algorithm:
        1. Filter entities by confidence threshold
        2. Sort entities by start position (descending)
        3. Process from end to start (prevents offset shifts)
        4. Replace each entity with mask value
        5. Return masked text

        Args:
            text: Original text containing PII
            entities: List of detected PII entities (from PIIDetector)

        Returns:
            Text with PII masked according to strategy
        """
        if not text or not entities:
            return text

        # Filter by confidence threshold
        filtered = [
            e for e in entities
            if e.confidence >= self.confidence_threshold
        ]

        if not filtered:
            return text

        # Sort by start position (descending) to process from end to start
        sorted_entities = sorted(
            filtered,
            key=lambda e: e.start,
            reverse=True
        )

        # Process each entity
        masked_text = text
        for entity in sorted_entities:
            mask_value = self.get_mask_value(entity)
            # Replace text from start to end with mask value
            masked_text = (
                masked_text[:entity.start] +
                mask_value +
                masked_text[entity.end:]
            )

        logger.debug(
            f"Masked {len(filtered)} entities using {self.strategy} strategy"
        )

        return masked_text

    def get_mask_value(self, entity: PIIEntity) -> str:
        """
        Get appropriate mask value for entity based on strategy.

        Args:
            entity: PII entity to mask

        Returns:
            Mask string to replace entity
        """
        if self.strategy == "placeholder":
            return self._placeholder_mask(entity)
        elif self.strategy == "partial":
            return self._partial_mask(entity)
        elif self.strategy == "hash":
            return self._hash_mask(entity)
        elif self.strategy == "replacement":
            return self._consistent_replacement(entity)
        else:
            # Fallback (shouldn't reach here due to __init__ validation)
            return self._placeholder_mask(entity)

    def apply_strategy(self, entity: PIIEntity) -> str:
        """
        Apply specific masking strategy.

        Args:
            entity: Entity to mask

        Returns:
            Masked value
        """
        return self.get_mask_value(entity)

    def preserve_format(self, entity: PIIEntity, masked: str) -> str:
        """
        Preserve original format (useful for numbers/dates).

        Maintains the length and character type of original.

        Args:
            entity: Original entity
            masked: Masked value

        Returns:
            Format-preserved masked value
        """
        original_len = len(entity.text)
        masked_len = len(masked)

        if masked_len == original_len:
            return masked

        # If masked is shorter, pad with * or =
        if masked_len < original_len:
            padding_char = "*" if entity.entity_type in [
                "CREDIT_CARD_MASKED",
                "CREDIT_CARD"
            ] else "="
            return masked + (padding_char * (original_len - masked_len))

        # If masked is longer, truncate
        return masked[:original_len]

    def _placeholder_mask(self, entity: PIIEntity) -> str:
        """
        Replace with [MASKED_ENTITY_TYPE].

        Example:
            "4532123456789010" → "[MASKED_CREDIT_CARD]"

        Args:
            entity: Entity to mask

        Returns:
            Placeholder mask
        """
        entity_type = entity.entity_type.replace("_", "")
        return f"[MASKED_{entity_type}]"

    def _partial_mask(self, entity: PIIEntity) -> str:
        """
        Show first 4 and last 4 characters.

        Example:
            "4532123456789010" → "4532****6789010"

        Args:
            entity: Entity to mask

        Returns:
            Partially masked value
        """
        text = entity.text
        if len(text) <= 8:
            # Too short to show first 4 + last 4, mask most of it
            return text[0] + ("*" * max(1, len(text) - 2)) + text[-1]

        first_4 = text[:4]
        last_4 = text[-4:]
        middle_len = len(text) - 8
        middle = "*" * middle_len

        return first_4 + middle + last_4

    def _hash_mask(self, entity: PIIEntity) -> str:
        """
        SHA256 hash with prefix.

        Example:
            "test@example.com" → "[SHA256:a1b2c3d4...]"

        Args:
            entity: Entity to mask

        Returns:
            Hash-based mask with prefix
        """
        text = entity.text
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()
        # Show first 8 characters of hash
        return f"[SHA256:{hash_hex[:8]}]"

    def _consistent_replacement(self, entity: PIIEntity) -> str:
        """
        Maintain consistency for same values across message.

        Example:
            First "TXN001" → "[TXN:ANON_001]"
            Second "TXN001" → "[TXN:ANON_001]" (same)

        Args:
            entity: Entity to mask

        Returns:
            Consistently masked value
        """
        text = entity.text
        entity_type = entity.entity_type

        # Check if we've seen this value before
        key = (entity_type, text)

        if key not in self.replacement_map:
            # Generate new token
            self.token_counter += 1
            type_prefix = entity_type.split("_")[0]
            token = f"{type_prefix}:ANON_{self.token_counter:06d}"
            self.replacement_map[key] = f"[{token}]"

        return self.replacement_map[key]

    def reset_replacement_map(self) -> None:
        """
        Clear replacement mapping and reset counter.

        Use this when starting a new masking session.
        """
        self.replacement_map.clear()
        self.token_counter = 0
        logger.debug("Replacement map reset")

    def get_replacement_map(self) -> Dict[str, str]:
        """
        Get current replacement mapping (for debugging/auditing).

        Returns:
            Dictionary of original values to masked values
        """
        return self.replacement_map.copy()

    def mask_batch(
        self,
        texts: List[str],
        entities_list: List[List[PIIEntity]]
    ) -> List[str]:
        """
        Mask multiple texts efficiently.

        Args:
            texts: List of texts to mask
            entities_list: List of entity lists (one per text)

        Returns:
            List of masked texts
        """
        return [
            self.mask_text(text, entities)
            for text, entities in zip(texts, entities_list)
        ]
