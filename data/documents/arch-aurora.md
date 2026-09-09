---
doc_id: arch-aurora
title: Aurora Architecture Overview
category: architecture
tags: aurora,ingestion,rate-limiting
---
Aurora is Nimbus's event ingestion service, owned by team Nimbus-Core. Clients
send events over HTTP or gRPC; Aurora buffers incoming events in an in-memory
ring buffer before handing them off to Cascade for downstream processing.

To protect the ring buffer from a single noisy client, Aurora enforces a
default rate limit of 5,000 events per second per tenant. Requests that
exceed this limit receive an HTTP 429 response with a Retry-After header.
The limit is configurable per tenant but requires a change request.

Aurora tracks ring buffer occupancy as a percentage; sustained occupancy
above 80% is considered a warning sign of an upstream slowdown or a tenant
exceeding its allotted throughput, and is the leading indicator used in the
Aurora throttling runbook.
