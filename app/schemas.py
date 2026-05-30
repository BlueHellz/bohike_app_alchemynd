from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class EntropyCurve(BaseModel):
    initial_hz: float = Field(..., ge=0.5, le=40.0)
    target_hz: float = Field(..., ge=0.5, le=40.0)
    curve: Literal["linear", "ease_in", "ease_out", "ease_in_out", "custom"] = "ease_in_out"
    custom_points: Optional[List[dict]] = None


class AudioConfig(BaseModel):
    binaural_carrier_hz: float = 200.0
    isochronic_intensity: float = Field(0.4, ge=0.0, le=1.0)
    noise_types: List[Literal["pink", "brown", "white", "violet", "black", "grey"]]
    noise_blend: List[float]
    ambient_type: str = "dawn_forest"
    ambient_intensity: float = Field(0.8, ge=0.0, le=1.0)


class MeditationConfig(BaseModel):
    enabled: bool = True
    style: Literal[
        "guided", "breathwork", "body_scan", "nlp_reframe",
        "hypnotic_induction", "visualization", "silent"
    ] = "guided"
    voice_tone: Literal["calm", "warm", "authoritative", "whispering"] = "calm"
    pacing: Literal["slow", "medium", "fast"] = "slow"
    script_outline: str = ""


class SubliminalConfig(BaseModel):
    enabled: bool = False
    messages: List[str] = []


class MobileVisual(BaseModel):
    color_hex: str
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


class SessionBlueprint(BaseModel):
    user_id: str
    client_type: Literal["mobile", "vr"]
    duration_seconds: int = Field(..., ge=60, le=10800)
    intent_summary: str = ""
    entrainment: EntropyCurve
    audio: AudioConfig
    meditation: MeditationConfig
    subliminal: SubliminalConfig
    mobile_visual: MobileVisual
    vr_visual: Optional[VRVisual] = None
    active_modalities: List[str] = []


class DesignRequest(BaseModel):
    user_text: str
    user_id: str
    client_type: Optional[Literal["mobile", "vr"]] = None
    preferred_modalities: Optional[List[str]] = None


class DesignResponse(BaseModel):
    blueprint: Optional[SessionBlueprint] = None
    requires_duration: bool = False
    safety_summary: str = ""


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    notes: Optional[str] = None
