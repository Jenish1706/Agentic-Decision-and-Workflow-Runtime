---
doc_id: api-anchor
title: API Reference: Anchor Auth
category: api_reference
tags: anchor,api,tokens
---
POST /v1/token — exchanges client credentials for a new access token (15
minute TTL) and refresh token (30 day TTL).

POST /v1/token/refresh — exchanges a valid refresh token for a new access
token without re-authenticating.

POST /v1/token/introspect — used internally by Compass to validate a
caller's access token; returns whether the token is active and its
associated scopes.

Access tokens are not revocable before expiry; because the TTL is only 15
minutes, Anchor relies on natural expiry rather than a revocation list.
