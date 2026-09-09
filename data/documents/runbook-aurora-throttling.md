---
doc_id: runbook-aurora-throttling
title: Runbook: Aurora Throttling / 429s
category: runbook
tags: aurora,throttling,incident
---
Use this runbook when Aurora starts returning 429 responses to a tenant, or
when Beacon alerts that Aurora's ring buffer occupancy is above 80%.

Steps:
1. Check the tenant's current request rate against its configured limit
   (default 5,000 events/sec/tenant).
2. Check ring buffer occupancy. If it is climbing even though the offending
   tenant is within its limit, the issue may be a downstream slowdown in
   Cascade rather than the tenant itself.
3. If the tenant has a legitimate, temporary need for a higher limit, raise
   the per-tenant limit via the Aurora config and notify the tenant.
4. If occupancy remains high after addressing the tenant's rate, escalate to
   the Nimbus-Core on-call engineer following the on-call escalation runbook.
