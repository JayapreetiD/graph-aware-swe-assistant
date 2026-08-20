

"""
llm/citation_verifier.py

Responsibility: parse citations out of an LLM-generated answer and
verify each one against the chunks that were actually supplied in
the prompt's context.

This is what makes "citation correctness" (a Phase 5 metric) an
objective, checkable number instead of a manual eyeball judgment.
It does not call the LLM and does not build prompts - pure
post-processing of (answer_text, chunks_used) -> verification report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches the exact format we instruct the LLM to use in
# prompt_builder.py's SYSTEM_INSTRUCTIONS:
#   (file:start_line-end_line)
# Example: (click/core.py:120-145)
#
# Kept intentionally strict (not fuzzy) - if the LLM drifts from this
# format, that's a signal worth surfacing (instruction-following
# failure), not something to silently paper over with a looser regex.
CITATION_PATTERN = re.compile(
    r"\(([^\s():]+):(\d+)-(\d+)\)"
)


@dataclass
class ParsedCitation:
    raw_text: str
    filepath: str
    start_line: int
    end_line: int


@dataclass
class CitationVerdict:
    citation: ParsedCitation
    is_valid: bool
    reason: str  # "matches_chunk" | "unknown_file" | "line_range_mismatch"


@dataclass
class VerificationReport:
    total_citations: int
    valid_citations: int
    invalid_citations: int
    verdicts: list[CitationVerdict] = field(default_factory=list)

    @property
    def citation_correctness_rate(self) -> float:
        """
        Fraction of citations that are grounded in the supplied
        context. Returns 1.0 if there were zero citations to check -
        callers should check total_citations separately if "no
        citations at all" is itself a failure mode worth flagging
        (e.g. the LLM answered without citing anything).
        """
        if self.total_citations == 0:
            return 1.0
        return self.valid_citations / self.total_citations


def parse_citations(answer_text: str) -> list[ParsedCitation]:
    """Extract every (file:start-end) citation from the answer text."""
    citations = []
    for match in CITATION_PATTERN.finditer(answer_text):
        raw, filepath, start_str, end_str = match.group(0), *match.groups()
        citations.append(
            ParsedCitation(
                raw_text=raw,
                filepath=filepath,
                start_line=int(start_str),
                end_line=int(end_str),
            )
        )
    return citations


def _verify_single_citation(
    citation: ParsedCitation, chunks_used: list[dict]
) -> CitationVerdict:
    """
    A citation is valid only if BOTH the filepath matches a chunk
    AND the cited line range falls within that chunk's actual line
    range. Matching filepath alone is not enough - the LLM could cite
    a real file with a fabricated line range, which is a subtler
    hallucination than an unknown file.
    """
    matching_file_chunks = [
        c for c in chunks_used if c["filepath"] == citation.filepath
    ]

    if not matching_file_chunks:
        return CitationVerdict(
            citation=citation, is_valid=False, reason="unknown_file"
        )

    for chunk in matching_file_chunks:
        # Citation is valid if it falls within (or exactly matches)
        # a retrieved chunk's line range. Using containment rather
        # than exact match, since the LLM may cite a sub-range of a
        # larger chunk (e.g. one method within a class chunk).
        if (
            chunk["start_line"] <= citation.start_line
            and citation.end_line <= chunk["end_line"]
        ):
            return CitationVerdict(
                citation=citation, is_valid=True, reason="matches_chunk"
            )

    return CitationVerdict(
        citation=citation, is_valid=False, reason="line_range_mismatch"
    )


def verify_citations(
    answer_text: str, chunks_used: list[dict]
) -> VerificationReport:
    """
    Main entry point: parse all citations from an answer and verify
    each against the chunks that were actually in context.

    chunks_used should be exactly PromptResult.chunks_used from
    prompt_builder.py - the ground truth of what the LLM was actually
    shown, not the full retrieval result (which may have been
    truncated by the token budget before reaching the LLM).
    """
    citations = parse_citations(answer_text)
    verdicts = [_verify_single_citation(c, chunks_used) for c in citations]

    valid_count = sum(1 for v in verdicts if v.is_valid)

    return VerificationReport(
        total_citations=len(citations),
        valid_citations=valid_count,
        invalid_citations=len(citations) - valid_count,
        verdicts=verdicts,
    )


if __name__ == "__main__":
    # Smoke test: one valid citation (matches a chunk), one invalid
    # (unknown file), one invalid (right file, wrong line range).
    fake_chunks = [
        {
            "node_id": "click/core.py::Command.invoke",
            "filepath": "click/core.py",
            "start_line": 100,
            "end_line": 150,
        },
    ]

    fake_answer = (
        "The invoke method handles command execution (click/core.py:100-150). "
        "It also touches config loading (click/config.py:10-20). "
        "And references line ranges outside the chunk (click/core.py:200-210)."
    )

    report = verify_citations(fake_answer, fake_chunks)

    print(f"Total citations: {report.total_citations}")
    print(f"Valid: {report.valid_citations}")
    print(f"Invalid: {report.invalid_citations}")
    print(f"Correctness rate: {report.citation_correctness_rate:.2f}")
    print()
    for v in report.verdicts:
        print(f"  {v.citation.raw_text} -> valid={v.is_valid} ({v.reason})")