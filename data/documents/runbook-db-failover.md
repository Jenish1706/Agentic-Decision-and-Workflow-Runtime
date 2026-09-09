---
doc_id: runbook-db-failover
title: Runbook: Database Failover
category: runbook
tags: cascade,anchor,database,failover
---
This runbook covers failing over a Postgres primary to its replica for
either Cascade or Anchor, the two services backed by Postgres.

Expected recovery targets: Recovery Time Objective (RTO) of 5 minutes and
Recovery Point Objective (RPO) of 30 seconds. If a failover is taking
noticeably longer than 5 minutes, escalate immediately rather than waiting.

Steps:
1. Confirm the primary is actually unreachable (not just slow) using Beacon.
2. Promote the replica using the database tool's `promote` command.
3. Update the service's connection configuration to point at the new
   primary, if not handled automatically by service discovery.
4. Verify writes are succeeding again before declaring the failover
   complete.
5. The old primary, once recovered, rejoins the cluster as a replica; it
   must not be reintroduced as primary without a manual review.
