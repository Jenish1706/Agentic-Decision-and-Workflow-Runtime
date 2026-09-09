---
doc_id: arch-ferry
title: Ferry Architecture Overview
category: architecture
tags: ferry,storage,objects
---
Ferry is Nimbus's object storage service, owned by team Nimbus-Data. It
stores blobs with configurable per-bucket retention policies and supports
object versioning. Ferry uses tiered storage: recently written or frequently
accessed objects live in a "hot" tier, while older, rarely accessed objects
are migrated to a cheaper "cold" tier by a background job.

Buckets have a storage quota; once a bucket approaches its quota, the
storage cleanup runbook describes how to reclaim space, either by migrating
more aggressively to cold storage or by deleting objects that are no longer
needed.
