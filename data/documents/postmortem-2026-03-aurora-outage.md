---
doc_id: postmortem-2026-03-aurora-outage
title: Postmortem: Aurora Ring Buffer Overflow (March 2026)
category: postmortem
tags: aurora,incident,postmortem
---
On March 2026, Aurora's ring buffer overflowed during a sudden traffic spike
from a single tenant, causing ingestion latency to climb to roughly 40
seconds for all tenants sharing that buffer shard.

Root cause: a recent deploy had introduced a second code path for event
ingestion that did not enforce the per-tenant rate cap, allowing the tenant
to exceed the normal 5,000 events/sec/tenant limit undetected.

Resolution: the missing rate cap was re-enabled on the affected path, and a
new Beacon alert was added to page on-call whenever ring buffer occupancy
exceeds 80%, rather than waiting for customer-visible latency to appear.

Follow-up action: all new ingestion code paths must now go through a shared
rate-limiting middleware instead of implementing their own checks.
