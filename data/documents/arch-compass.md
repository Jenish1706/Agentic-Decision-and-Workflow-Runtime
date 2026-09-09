---
doc_id: arch-compass
title: Compass Architecture Overview
category: architecture
tags: compass,gateway,routing
---
Compass is Nimbus's API gateway, owned by team Nimbus-Core. Every external
request to a Nimbus service is routed through Compass, which terminates TLS,
applies a global default rate limit of 10,000 requests per second, and
forwards the request to the appropriate backend service based on path-based
routing rules.

Before forwarding a request, Compass validates the caller's access token by
calling Anchor's token introspection endpoint. If the token is missing,
expired, or invalid, Compass rejects the request with a 401 before it ever
reaches the backend service. This means backend services generally do not
need to re-implement authentication themselves.
