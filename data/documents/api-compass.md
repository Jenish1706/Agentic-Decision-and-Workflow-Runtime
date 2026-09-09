---
doc_id: api-compass
title: API Reference: Compass Gateway
category: api_reference
tags: compass,api,gateway
---
Compass does not expose its own business-logic endpoints; it forwards
requests to backend services based on path prefix (for example, requests to
`/aurora/*` are forwarded to Aurora).

Compass adds an `X-Nimbus-Caller` header identifying the authenticated
caller before forwarding, and strips any client-supplied header with that
same name to prevent spoofing. All forwarded requests carry the gateway's
global rate limit headers as well as any service-specific limit headers
returned by the backend.
