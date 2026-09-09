---
doc_id: faq-token-expiry
title: FAQ: Why Did My Token Stop Working
category: faq
tags: anchor,tokens,faq
---
Q: My access token stopped working after about 15 minutes. Is that a bug?
A: No, this is expected. Anchor issues access tokens with a 15 minute TTL by
design. Use your refresh token (valid for 30 days) to obtain a new access
token instead of re-authenticating from scratch.

Q: What happens when my refresh token itself expires after 30 days?
A: You must re-authenticate with Anchor from scratch to obtain a new
refresh token and access token pair.
