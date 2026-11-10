"""Minimal-pair schema: one row per fused string, one junction per join (docs/data-format.md)."""

from enum import StrEnum
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, computed_field


class RuleClass(StrEnum):
    VOWEL = "vowel"
    VISARGA = "visarga"
    CONSONANT = "consonant"
    NONE = "none"


class RuleId(StrEnum):
    GUNA = "guna"
    SAVARNA_DIRGHA = "savarna_dirgha"
    YAN = "yan"
    WORD_FINAL_VOICING = "word_final_voicing"
    VISARGA_S = "visarga_s"
    VISARGA_O = "visarga_o"
    VISARGA_O_AVAGRAHA = "visarga_o_avagraha"
    VISARGA_DROP = "visarga_drop"


class Rule(NamedTuple):
    group: RuleClass
    sutra: str  # Pāṇini, Aṣṭādhyāyī numbering


# grows as extraction covers more rules
RULES: dict[RuleId, Rule] = {
    RuleId.GUNA: Rule(RuleClass.VOWEL, "6.1.87"),
    RuleId.SAVARNA_DIRGHA: Rule(RuleClass.VOWEL, "6.1.101"),
    RuleId.YAN: Rule(RuleClass.VOWEL, "6.1.77"),
    RuleId.WORD_FINAL_VOICING: Rule(RuleClass.CONSONANT, "8.2.39"),
    RuleId.VISARGA_S: Rule(RuleClass.VISARGA, "8.3.34"),
    RuleId.VISARGA_O: Rule(RuleClass.VISARGA, "6.1.114"),
    RuleId.VISARGA_O_AVAGRAHA: Rule(RuleClass.VISARGA, "6.1.113"),
    RuleId.VISARGA_DROP: Rule(RuleClass.VISARGA, "8.3.19"),
}


class BoundaryLevel(StrEnum):
    INTACT = "intact"
    CHANGED = "changed"
    MERGED = "merged"


class Junction(BaseModel):
    span: tuple[int, int]  # half-open [start, end) over fused; zero-width when nothing changed
    cut: int | None  # where the left word ends in fused; None when a letter belongs to both sides
    underlying_left: str
    underlying_right: str
    surface: str
    rule_class: RuleClass
    rule_ids: list[RuleId] = []
    type: Literal["compound", "external"]

    @computed_field
    @property
    def boundary_level(self) -> BoundaryLevel:
        if self.underlying_left + self.underlying_right == self.surface:
            return BoundaryLevel.INTACT
        if self.cut is None:
            return BoundaryLevel.MERGED
        return BoundaryLevel.CHANGED


class Pair(BaseModel):
    id: str
    fused: str  # keeps the space for sandhi across words, e.g. "rAma udakam"
    split_parts: list[str] = Field(min_length=2)
    alt_splits: list[list[str]] = []  # other valid splits; junctions describe split_parts only
    junctions: list[Junction]
    encoding: Literal["slp1"] = "slp1"
    source_corpus: Literal["DCS", "UoH"]
    source_ref: str
    sentence_id: str
    split_provenance: Literal["gold", "reconstructed", "derived"]
    frequency: int = Field(ge=0)
    frequency_bucket: Literal["high", "mid", "low"] | None = None  # set once cutoffs are fixed
    is_control: bool = False
    split_assignment: Literal["train", "dev", "test"]
