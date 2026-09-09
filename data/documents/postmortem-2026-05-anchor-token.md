---
doc_id: postmortem-2026-05-anchor-token
title: Postmortem: Anchor Token TTL Misconfiguration (May 2026)
category: postmortem
tags: anchor,incident,postmortem
---
On May 2026, a configuration typo caused Anchor to issue access tokens with
a 24 hour TTL instead of the intended 15 minutes.

The issue was caught within 40 minutes by an automated config-drift check
and reverted immediately. A security audit of tokens issued during the
window found no evidence of misuse.

Follow-up action: the config-drift check that caught this was previously
advisory-only (logged but did not page); it now pages the Nimbus-Platform
on-call engineer immediately when token TTL configuration drifts from the
approved value.
