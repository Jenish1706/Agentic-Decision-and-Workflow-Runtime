---
doc_id: api-aurora
title: API Reference: Aurora Ingestion
category: api_reference
tags: aurora,api,ingestion
---
POST /v1/events

Ingests a single event. Requires an Authorization header with a valid Anchor
access token. The request body must include `tenant_id`, `event_type`, and
a `payload` object.

Responses:
- 202 Accepted: event was buffered successfully.
- 413 Payload Too Large: payload exceeds the 256KB limit.
- 429 Too Many Requests: tenant exceeded its rate limit (default 5,000
  events/sec/tenant). The response includes a Retry-After header.

Successful responses include an `X-RateLimit-Remaining` header indicating
how much of the tenant's per-second budget is left.
