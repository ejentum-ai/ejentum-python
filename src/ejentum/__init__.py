"""
Ejentum Python SDK - Deterministic Reasoning-Augmented Retrieval (RAR)
"""

from .client import Ejentum
from .wrappers import wrap_openai, wrap_anthropic
from .exceptions import APIError, AuthenticationError, EjentumError, EjentumSecurityError, EvidenceVerificationResult, ScaffoldNotFoundError
from .checkpoints import (
    BaseCheckpointStore,
    MemoryCheckpointStore,
    FileCheckpointStore,
    RedisCheckpointStore,
    S3CheckpointStore,
    GCSCheckpointStore,
    create_checkpoint_store,
)
from .handshake import (
    AgentRole,
    HandshakeStatus,
    HandshakeContract,
    HandshakeResult,
    MultiAgentTopology,
    HandshakeEngine,
    get_preset_topology,
    list_preset_topologies,
)
from .visual import (
    BoundingBox,
    SpatialLayoutVerifier,
    SpatialVerificationResult,
    DiagramArtifactTrap,
    DiagramVerificationResult,
    FrameSequenceDriftGuard,
    FrameSequenceResult,
)
from .native import is_native_available, get_native_version, fast_iou, fast_directional
from .temporal import (
    TemporalGuard,
    TemporalPolicyRule,
    RequirePriorRule,
    ForbidFollowingRule,
    RequireParamConditionRule,
    RequireApprovalRule,
    TemporalVerificationResult,
    ActionRecord,
)
from .nli import LocalNLIEngine, NLIResult
from .adaptive import AdaptiveRouter, AdaptiveRouteResult, VerificationTier
from .schema import StructuredOutputEngine, StructuredCoercionResult
from . import integrations
from .integrations import (
    EjentumGuardNode,
    EjentumConditionalEdge,
    EjentumHandshakeNode,
    EjentumVisualGuardNode,
    EjentumStepHook,
    EjentumGuardTool,
    EjentumHandshakeHook,
)

__version__ = "1.8.0"
__all__ = [
    "Ejentum",
    "wrap_openai",
    "wrap_anthropic",
    "EjentumError",
    "AuthenticationError",
    "APIError",
    "ScaffoldNotFoundError",
    "EjentumSecurityError",
    "EvidenceVerificationResult",
    "BaseCheckpointStore",
    "MemoryCheckpointStore",
    "FileCheckpointStore",
    "RedisCheckpointStore",
    "S3CheckpointStore",
    "GCSCheckpointStore",
    "create_checkpoint_store",
    "AgentRole",
    "HandshakeStatus",
    "HandshakeContract",
    "HandshakeResult",
    "MultiAgentTopology",
    "HandshakeEngine",
    "get_preset_topology",
    "list_preset_topologies",
    "BoundingBox",
    "SpatialLayoutVerifier",
    "SpatialVerificationResult",
    "DiagramArtifactTrap",
    "DiagramVerificationResult",
    "FrameSequenceDriftGuard",
    "FrameSequenceResult",
    "is_native_available",
    "get_native_version",
    "fast_iou",
    "fast_directional",
    "TemporalGuard",
    "TemporalPolicyRule",
    "RequirePriorRule",
    "ForbidFollowingRule",
    "RequireParamConditionRule",
    "RequireApprovalRule",
    "TemporalVerificationResult",
    "ActionRecord",
    "LocalNLIEngine",
    "NLIResult",
    "AdaptiveRouter",
    "AdaptiveRouteResult",
    "VerificationTier",
    "StructuredOutputEngine",
    "StructuredCoercionResult",
    "integrations",
    "EjentumGuardNode",
    "EjentumConditionalEdge",
    "EjentumHandshakeNode",
    "EjentumVisualGuardNode",
    "EjentumStepHook",
    "EjentumGuardTool",
    "EjentumHandshakeHook",
]


