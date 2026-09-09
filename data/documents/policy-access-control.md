---
doc_id: policy-access-control
title: Policy: Access Control
category: policy
tags: access,security,policy
---
Access to production systems requires both multi-factor authentication and
an active VPN connection; neither alone is sufficient.

"Break-glass" emergency access (used when normal access paths are
unavailable during an incident) is logged automatically and reviewed by a
security lead every week, regardless of whether it was actually used for an
emergency.

All service-to-service authentication must use Anchor-issued tokens; direct
service-to-service credentials (shared secrets, static API keys) are not
permitted for new services.
