"""Pydantic models for vLLM Semantic Router configuration."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Listener(BaseModel):
    """Network listener configuration."""

    name: str
    address: str
    port: int
    timeout: Optional[str] = "300s"


class KeywordSignal(BaseModel):
    """Keyword-based signal configuration."""

    name: str
    operator: str
    keywords: List[str]
    case_sensitive: bool = False


class EmbeddingSignal(BaseModel):
    """Embedding-based signal configuration."""

    name: str
    threshold: float
    candidates: List[str]
    aggregation_method: str = "max"


class Domain(BaseModel):
    """Domain category configuration."""

    name: str
    description: str
    mmlu_categories: Optional[List[str]] = None


class FactCheck(BaseModel):
    """Fact-checking signal configuration."""

    name: str
    description: str


class UserFeedback(BaseModel):
    """User feedback signal configuration."""

    name: str
    description: str


class Preference(BaseModel):
    """Route preference signal configuration."""

    name: str
    description: str


class Signals(BaseModel):
    """All signal configurations."""

    keywords: Optional[List[KeywordSignal]] = []
    embeddings: Optional[List[EmbeddingSignal]] = []
    domains: Optional[List[Domain]] = []
    fact_check: Optional[List[FactCheck]] = []
    user_feedbacks: Optional[List[UserFeedback]] = []
    preferences: Optional[List[Preference]] = []


class Condition(BaseModel):
    """Routing condition."""

    type: str
    name: str


class Rules(BaseModel):
    """Routing rules."""

    operator: str
    conditions: List[Condition]


class ModelRef(BaseModel):
    """Model reference in decision."""

    model: str
    use_reasoning: Optional[bool] = False


class PluginConfig(BaseModel):
    """Plugin configuration."""

    type: str
    configuration: Dict[str, Any]


class Decision(BaseModel):
    """Routing decision configuration."""

    name: str
    description: str
    priority: int
    rules: Rules
    modelRefs: List[ModelRef] = Field(alias="modelRefs")
    plugins: Optional[List[PluginConfig]] = []

    class Config:
        populate_by_name = True


class Endpoint(BaseModel):
    """Backend endpoint configuration."""

    name: str
    weight: int
    endpoint: str
    protocol: str = "http"


class ModelPricing(BaseModel):
    """Model pricing configuration."""

    currency: Optional[str] = "USD"
    prompt_per_1m: Optional[float] = 0.0
    completion_per_1m: Optional[float] = 0.0


class Model(BaseModel):
    """Model configuration."""

    name: str
    endpoints: List[Endpoint]
    access_key: Optional[str] = None
    reasoning_family: Optional[str] = None
    pricing: Optional[ModelPricing] = None


class ReasoningFamily(BaseModel):
    """Reasoning family configuration."""

    type: str
    parameter: str


class ExternalModel(BaseModel):
    """External model configuration."""

    role: str  # "preference", "guardrail", etc.
    provider: str  # "vllm"
    endpoint: str  # "host:port"
    model_name: str
    timeout_seconds: Optional[int] = 30
    parser_type: Optional[str] = "json"
    access_key: Optional[str] = None  # Optional access key for Authorization header


class Providers(BaseModel):
    """Provider configuration."""

    models: List[Model]
    default_model: Optional[str] = None
    reasoning_families: Optional[Dict[str, ReasoningFamily]] = {}
    default_reasoning_effort: Optional[str] = "high"
    external_models: Optional[List[ExternalModel]] = []


class MemoryMilvusConfig(BaseModel):
    """Milvus configuration for memory storage."""

    address: str
    collection: str = "agentic_memory"
    dimension: int = 384


class MemoryEmbeddingConfig(BaseModel):
    """Embedding configuration for memory."""

    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class MemoryQueryRewriteConfig(BaseModel):
    """Query rewrite configuration for memory search."""

    enabled: bool = False
    endpoint: Optional[str] = None
    model: Optional[str] = None


class MemoryExtractionConfig(BaseModel):
    """Fact extraction configuration for memory saving."""

    enabled: bool = True
    endpoint: str
    model: str
    batch_size: int = 10  # Extract every N turns (must match Go config field name)
    timeout_seconds: int = 30  # Timeout for LLM extraction calls


class MemoryConfig(BaseModel):
    """Agentic Memory configuration for cross-session memory."""

    enabled: bool = True
    milvus: Optional[MemoryMilvusConfig] = None
    embedding: Optional[MemoryEmbeddingConfig] = None
    default_retrieval_limit: int = 5
    default_similarity_threshold: float = 0.70
    query_rewrite: Optional[MemoryQueryRewriteConfig] = None
    extraction: Optional[MemoryExtractionConfig] = None


class UserConfig(BaseModel):
    """Complete user configuration."""

    version: str
    listeners: List[Listener]
    signals: Optional[Signals] = None
    decisions: List[Decision]
    providers: Providers
    memory: Optional[MemoryConfig] = None  # Agentic Memory config

    class Config:
        populate_by_name = True
