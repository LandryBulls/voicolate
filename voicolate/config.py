"""Configuration for the v3 cross-microphone isolation path.

Every parameter that affects an output lives here, so a run can be described by a
single serialisable object and written next to the audio it produced. The older
``isolate()`` path scatters its parameters across three entry-point scripts with
divergent defaults, which is why existing ``_isolated_v2.wav`` files cannot be
traced back to the settings that made them.
"""

from dataclasses import dataclass, asdict, field
from typing import Tuple


@dataclass
class IsolationConfig:
    # --- STFT ---
    rate: int = 44100
    nperseg: int = 2048          # ~46 ms at 44.1 kHz
    noverlap: int = 1536         # hop 512 -> ~11.6 ms frames

    # --- per-track calibration ---
    # 'speech_level'   equalise the 95th-percentile frame level (talk-time invariant)
    # 'bleed_symmetry' solve for gains that symmetrise the measured bleed matrix
    # 'rms'            legacy: equalise overall RMS (biased by how much each person talked)
    # 'none'           leave levels as recorded
    calibration: str = 'speech_level'
    target_level_db: float = -20.0
    noise_floor_percentile: float = 5.0
    speech_level_percentile: float = 95.0

    # --- time-frequency dominance mask ---
    dominance_margin_db: float = 2.0    # dB advantage at which the mask is half open
    dominance_width_db: float = 6.0     # 10-90% transition width, in dB of dominance
    mask_floor: float = 0.06            # never fully close a bin (~-24 dB)
    smoothing_freq_bins: int = 5
    smoothing_time_frames: int = 3
    harmonicity_weight: float = 0.0     # broadband flatness term; off by default

    # --- frame-level speech presence (replaces the peak-referenced soft gate) ---
    speech_band_hz: Tuple[float, float] = (100.0, 8000.0)
    presence_dominance_width_db: float = 3.0
    presence_snr_db: float = 8.0        # dB above this track's own noise floor
    presence_snr_width_db: float = 6.0
    hangover_ms: float = 250.0          # symmetric: protects onsets and unvoiced offsets
    residual_floor_db: float = -20.0    # hard limit on total attenuation; never silence

    # --- comfort-noise substitution (opt-in) ---
    # Floored suppression leaves the interfering voice quiet but structurally intact.
    # Setting this fills the suppressed regions with noise shaped like the
    # microphone's own ambience, burying the residual without punching holes in the
    # signal, weighted so nothing is ever added over the target's own speech.
    #
    # Off by default because it measures badly for transcription. On
    # 2024-05-22_000 TRACK01, against a WhisperX large-v2 baseline that repeats
    # bit-identically on the same audio:
    #
    #   fill    own words kept   words over own-VAD silence   words matching TRACK02
    #   none        100.0%                 10                          6
    #   +3 dB        95.7%                  0                          4
    #   +6 dB        95.7%                  0                          4
    #   +12 dB       94.8%                  0                          4
    #
    # It removes every word transcribed over the target's own silence, but costs
    # ~4.3% of that speaker's real words to do it -- and the cost is a step, not a
    # slope, so it comes from adding any noise at all rather than from its level.
    # Worth enabling when a stray word attributed to the wrong speaker is more
    # costly than a missing one, or for human listening; otherwise leave it off.
    residual_noise_over_db: float = None   # dB above the residual it is burying
    ambience_seconds: float = 20.0        # quiet audio sampled to shape the noise
    seed: int = 0

    # --- output ---
    output_level_db: float = -20.0      # one gain for the whole session, speech-referenced
    peak_limit_db: float = -1.0

    # --- execution ---
    block_seconds: float = 300.0
    block_pad_seconds: float = 1.0

    @property
    def hop(self) -> int:
        return self.nperseg - self.noverlap

    @property
    def residual_floor(self) -> float:
        return 10 ** (self.residual_floor_db / 20)

    @property
    def frame_rate(self) -> float:
        return self.rate / self.hop

    @property
    def fills_residual(self) -> bool:
        return self.residual_noise_over_db is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['speech_band_hz'] = list(d['speech_band_hz'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'IsolationConfig':
        d = dict(d)
        if 'speech_band_hz' in d:
            d['speech_band_hz'] = tuple(d['speech_band_hz'])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
