---
doc_id: faq-new-service-onboarding
title: FAQ: Onboarding a New Service
category: faq
tags: onboarding,faq
---
Q: What do I need to do to onboard a new internal service onto the Nimbus
platform?
A: Three things: register the service and its routing rules with Compass,
request client credentials from Anchor so the service can authenticate, and
set up basic Beacon alert rules (at minimum, an availability check) before
the service is considered production-ready.

Q: Is there a required order for these steps?
A: Register with Compass and Anchor first; Beacon alerting can be set up
once the service is actually deployed and emitting metrics.
