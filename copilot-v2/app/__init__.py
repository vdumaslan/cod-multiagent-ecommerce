"""
Seller copilot application package (pipeline, agents, API, precompute).

Grounding caches (pricing, sentiment, etc.) live under the shared artifacts root,
e.g. ``<artifacts_root>/caches/<snapshot_id>/pricing`` and ``.../sentiment``, not
under this package. See ``copilot_v2.scripts.build_pricing_cache`` /
``build_sentiment_cache`` defaults.
"""
