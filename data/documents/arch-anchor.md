---
doc_id: arch-anchor
title: Anchor Architecture Overview
category: architecture
tags: anchor,auth,tokens
---
Anchor is Nimbus's authentication and identity service, owned by team
Nimbus-Platform. It issues short-lived JSON Web Token (JWT) access tokens
with a 15 minute time-to-live, and longer-lived refresh tokens valid for 30
days. Access tokens are validated statelessly by services; refresh tokens
are checked against Anchor's session store, an in-memory cache with periodic
snapshots to disk for recovery.

Because access tokens expire after only 15 minutes, any client that expects
to make requests over a longer session must use its refresh token to obtain
a new access token before the old one expires.
