from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class EntertainmentCurve(BaseModel):
    initial_freq_hz: float = Field(..., ge=0.5, le=40.0)
    target_freq_hz: float = Field(..., ge=0.5, le=40.0)
    transition_curve: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "ease_in_out"
    custom_points: Optional[List[dict]] = None  # [{t_norm: 0.0, freq: 7.5}, ...]


class AudioScape(BaseModel):
    ambient_type: str  # "dawn_forest", "cosmic_drone", etc.
    intensity: float = Field(..., ge=0.0, le=1.0)
    binaural_base_freq_hz: float = 200.0
    isochronic_intensity: float = Field(0.4, ge=0.0, le=1.0)
    spatial_audio_model: Literal["stereo", "ambisonics"] = "stereo"


class MobileVisual(BaseModel):
    color_hex: str  # e.g. "#FFA500"
    pulse_sync: bool = True
    pulse_shape: Literal["sine", "none"] = "sine"


class VRElement(BaseModel):
    type: str
    sync_to: str
    params: dict


class VRVisual(BaseModel):
    scene_type: str
    color_palette: List[str]
    dynamic_elements: List[VRElement]
    global_light_rhythm: bool = True


class SubliminalConfig(BaseModel):
    enabled: bool = False
    messages: List[str] = []


class NeuroCognitiveIntent(BaseModel):
    cognitive_targets: List[str]
    emotional_targets: List[str]
    subconscious_goal: Optional[str] = None


class SessionBlueprint(BaseModel):
    user_id: str
    client_type: Literal["mobile", "vr"]
    duration_seconds: int = Field(..., ge=60, le=10800)
    entrainment: EntertainmentCurve
    audio_scape: AudioScape
    mobile_visual: MobileVisual
    vr_visual: Optional[VRVisual] = None
    subliminal: SubliminalConfig
    neuro_cognitive_intent: NeuroCognitiveIntent


class DesignRequest(BaseModel):
    user_text: str
    user_id: str
    client_type: Optional[Literal["mobile", "vr"]] = None


class DesignResponse(BaseModel):
    blueprint: Optional[SessionBlueprint] = None
    requires_duration: bool = False
    requires_affirmations: bool = False


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    notes: Optional[str] = None
