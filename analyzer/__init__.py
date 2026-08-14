"""Reference-video → style template analysis.

Use analyze_style() for a reference video (extracts editing STYLE) and
analyze_content() for the user's own footage (extracts CONTENT: duration,
scene changes, motion, audio peaks). The two extraction targets are kept
deliberately distinct — see pipeline.py.
"""

from .pipeline import analyze_content, analyze_style

__all__ = ["analyze_content", "analyze_style"]
