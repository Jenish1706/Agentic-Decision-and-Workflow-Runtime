---
doc_id: runbook-storage-cleanup
title: Runbook: Ferry Storage Cleanup
category: runbook
tags: ferry,storage,cleanup
---
Use this runbook when a Ferry bucket is approaching its storage quota.

Steps:
1. Check whether the bucket's cold-tier migration job is running
   successfully; a stalled migration job is the most common cause of
   unexpected quota pressure.
2. If migration is healthy but the bucket is still near quota, identify the
   largest and oldest objects as cold-storage migration candidates.
3. Deleting any object older than 30 days requires written approval from the
   bucket owner before it can be removed; this is a hard policy requirement,
   not a suggestion.
4. Once space is reclaimed, confirm the bucket has dropped below 80% of its
   quota before closing out the cleanup.
