---
doc_id: postmortem-2026-06-lighthouse-delay
title: Postmortem: Lighthouse Delivery Delay (June 2026)
category: postmortem
tags: lighthouse,incident,postmortem
---
On June 2026, an outage at Lighthouse's SMS delivery provider caused SMS
notifications, including on-call pages, to be delayed by up to 25 minutes.

Lighthouse's automatic retry logic kept attempting delivery through the
failing provider rather than switching providers, since at the time
Lighthouse only supported a single SMS provider per environment.

Resolution: delivery was manually failed over to a backup SMS provider,
and the dead-letter queue was drained once the primary provider recovered.

Follow-up action: Lighthouse now supports an automatic provider failover
after 2 consecutive delivery failures on the same channel, rather than
requiring a manual switch.
