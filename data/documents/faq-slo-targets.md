---
doc_id: faq-slo-targets
title: FAQ: SLO Targets by Tier
category: faq
tags: slo,tiers,faq
---
Q: What uptime SLO applies to a given service?
A: It depends on the service's tier. Tier 1 services target 99.9% uptime,
which allows roughly 43 minutes of downtime per month. Tier 2 services
target 99.5%, roughly 3.6 hours per month. Tier 3 services target 99%,
roughly 7.3 hours per month.

Q: How is a service's tier decided?
A: Tier is assigned based on how directly customer-facing and
business-critical the service is; Aurora, Compass, and Anchor are Tier 1
because most other services depend on them.
