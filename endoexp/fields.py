"""Closed-set report fields in the spirit of MST 3.0 (finding -> attribute -> value).

Every field is a small closed option set, so each one can be scored on its own
and read out as a distribution over options. Fields are grouped by what they
ask of the model:

- ``global``: scene-level facts (procedure, anatomy, bowel preparation);
- ``overlay``: burned-in artefacts that are shortcuts, not findings;
- ``descriptor``: local visual judgements about a finding (the main target);
- ``impression``: judgements that also need clinical knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

LETTERS = "ABCDEFGHIJ"
KINDS = ("global", "overlay", "descriptor", "impression")

GRID = ("upper-left", "upper-center", "upper-right",
        "center-left", "center", "center-right",
        "lower-left", "lower-center", "lower-right")


@dataclass(frozen=True)
class Field:
    name: str
    label: str            # line label in the structured report template
    question: str         # multiple-choice question
    options: tuple[str, ...]
    kind: str
    ordinal: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if len(set(self.options)) != len(self.options) or len(self.options) > len(LETTERS):
            raise ValueError(f"{self.name}: options must be unique and at most {len(LETTERS)}")

    def index(self, value: str) -> int:
        return self.options.index(value)


_FIELDS = (
    Field("procedure", "Procedure", "Which endoscopic procedure is this image taken from?",
          ("colonoscopy", "gastroscopy"), "global"),
    Field("landmark", "Anatomical landmark", "Which anatomical landmark is visible in this image?",
          ("none", "cecum", "ileum", "pylorus", "z-line", "retroflexed rectum", "retroflexed stomach"),
          "global"),
    Field("bbps", "Bowel preparation",
          "How would you rate the bowel preparation in this image on the Boston Bowel Preparation Scale?",
          ("0-1 (inadequate)", "2-3 (adequate)"), "global", ordinal=True),
    Field("text_overlay", "Text overlay", "Is there any text overlaid on this image?",
          ("no", "yes"), "overlay"),
    Field("box_overlay", "Box inset",
          "Is there a green or black box inset (such as a scope-position thumbnail) in a corner of this image?",
          ("no", "yes"), "overlay"),
    Field("abnormality", "Abnormality", "Which abnormality is visible in this image?",
          ("none", "polyp", "esophagitis", "ulcerative colitis", "Barrett's esophagus",
           "short-segment Barrett's esophagus", "hemorrhoids"), "descriptor"),
    Field("polyp_count", "Polyp count", "How many polyps are visible in this image?",
          ("0", "1", "2", "3 or more"), "descriptor", ordinal=True),
    Field("polyp_location", "Polyp location", "In which region of the image is the polyp centered?",
          GRID, "descriptor"),
    Field("polyp_extent", "Polyp extent", "How much of the image area does the polyp occupy?",
          ("less than 5%", "5% to 20%", "more than 20%"), "descriptor", ordinal=True),
    Field("polyp_size", "Polyp size", "What is the estimated size of the polyp?",
          ("smaller than 5 mm", "5 to 10 mm", "11 to 20 mm", "larger than 20 mm"),
          "descriptor", ordinal=True),
    Field("paris_type", "Paris classification",
          "What is the Paris classification of the polyp's morphology?",
          ("0-Ip (pedunculated)", "0-Is (sessile)", "0-IIa (slightly elevated)"), "descriptor"),
    Field("instrument", "Instrument", "Which instrument is visible in this image?",
          ("none", "tube", "metal clip", "polyp snare", "injection needle", "biopsy forceps"),
          "descriptor"),
    Field("instrument_count", "Instrument count", "How many instruments are visible in this image?",
          ("0", "1", "2 or more"), "descriptor", ordinal=True),
    Field("uc_grade", "Ulcerative colitis grade",
          "What is the endoscopic severity grade of the ulcerative colitis in this image?",
          ("grade 1 (mild)", "grade 2 (moderate)", "grade 3 (severe)"), "impression", ordinal=True),
    Field("esophagitis_grade", "Esophagitis grade",
          "What is the Los Angeles classification grade of the esophagitis in this image?",
          ("grade A", "grade B to D"), "impression", ordinal=True),
)

FIELDS: dict[str, Field] = {field.name: field for field in _FIELDS}

# Fields of the structured findings template, in report order (descriptors
# before impressions). Overlay fields are measured but not part of a report.
REPORT_FIELDS = ("procedure", "landmark", "bbps", "abnormality", "polyp_count",
                 "polyp_location", "polyp_extent", "polyp_size", "paris_type",
                 "instrument", "instrument_count", "uc_grade", "esophagitis_grade")


def get(name: str) -> Field:
    try:
        return FIELDS[name]
    except KeyError:
        raise KeyError(f"unknown field {name!r}; known: {sorted(FIELDS)}") from None


def grid_cell(cx: float, cy: float) -> str:
    """Name of the 3x3 cell containing a point in normalized [0, 1] coordinates."""
    col = min(int(cx * 3), 2)
    row = min(int(cy * 3), 2)
    return GRID[row * 3 + col]


def extent_bin(fraction: float) -> str:
    options = FIELDS["polyp_extent"].options
    if fraction < 0.05:
        return options[0]
    if fraction <= 0.20:
        return options[1]
    return options[2]
