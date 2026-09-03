"""Independent discovery sources behind one interface.

A single reverse-image endpoint is a wrapper, and it inherits that endpoint's blind
spots wholesale. Google Lens, for instance, deliberately does not do face recognition:
given a portrait it matches the eyewear, and given a face crop it returns strangers who
look similar. No amount of prompt or parameter tuning fixes that, because it is a policy
decision at the provider, not a quality problem.

So discovery fans out across sources that fail differently:

============ ============================================= ==================
Source        What it is good at                            Needs
============ ============================================= ==================
``lens``      the visual index: this exact photo, republished SERPAPI_KEY
``serpweb``   site-restricted keyword search on a name       SERPAPI_KEY
``exa``       neural retrieval; finds pages *about* a person  EXA_API_KEY
``wikidata``  structured, curated portraits of public figures nothing
============ ============================================= ==================

Every source returns :class:`~sigil.models.SearchCandidate` objects and nothing more.
None of them decides identity. They decide what gets *looked at*; the face gate in
:mod:`sigil.verify` decides what is true, and it applies the same calibrated threshold to
a Wikidata portrait and to a random visual match.

Sources are skipped silently when unconfigured, so the pipeline degrades to whatever
credentials are present rather than failing.
"""

from __future__ import annotations

from sigil.search.providers.base import ProviderResult, run_providers

__all__ = ["ProviderResult", "run_providers"]
