"""Single source of truth for the manual video review (used by the sheet generator and validator)."""
from __future__ import annotations

DECISIONS = ('include', 'exclude', 'reclassify')
DOMAINS = ('food', 'gaming', 'education')
PRIORITIES_NEEDING_NOTES = ('HIGH', 'MEDIUM')

DOMAIN_DEFINITIONS = {
    'food': 'Cooking, food preparation and food-focused content.',
    'gaming': 'Gameplay, game strategy, game reviews and direct gaming content.',
    'education': 'Structured teaching, academic lessons and demonstrable practical skill tutorials. '
                 'Excludes general celebrity news and entertainment without a meaningful educational purpose.',
}

# Columns the reviewer fills. Everything else in the sheet is reference information.
REVIEWER_COLUMNS = ('manual_review_decision', 'corrected_domain', 'reviewer_notes', 'reviewed_by', 'reviewed_at')

SHEET_COLUMNS = (
    'review_priority', 'video_id', 'video_title', 'channel_title', 'proposed_domain', 'open_video', 'youtube_url',
    'api_category_id', 'collected_comment_records', 'collected_top_level', 'api_comment_count',
    'original_verification_status', 'original_verified_by_AUTOMATED', 'original_verified_at_AUTOMATED',
    'original_verification_notes', 'assistant_flag_level', 'assistant_flag_reason',
) + REVIEWER_COLUMNS
