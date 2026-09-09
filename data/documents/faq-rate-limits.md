---
doc_id: faq-rate-limits
title: FAQ: Rate Limits
category: faq
tags: rate-limits,faq
---
Q: What are the default rate limits across Nimbus services?
A: Aurora defaults to 5,000 events per second per tenant. Compass, as the
shared gateway, defaults to a global limit of 10,000 requests per second
across all traffic passing through it.

Q: How do I request a higher limit for my tenant?
A: Submit a change request describing the expected traffic and duration;
temporary increases (for example, for a product launch) are generally
approved faster than permanent increases.
