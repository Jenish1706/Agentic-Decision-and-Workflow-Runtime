---
doc_id: arch-cascade
title: Cascade Architecture Overview
category: architecture
tags: cascade,etl,pipeline
---
Cascade is Nimbus's batch data pipeline, owned by team Nimbus-Data. It reads
events written by Aurora, runs a set of transformation jobs every 15 minutes,
and writes the transformed output to Ferry for long-term storage.

Cascade's job state and scheduling metadata are stored in a Postgres
database with a hot standby replica for failover. Because Cascade jobs run
on a 15 minute cycle, a short failover event is generally absorbed by the
next scheduled run without any manual intervention beyond what the failover
runbook describes.
