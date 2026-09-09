---
doc_id: faq-alert-noise
title: FAQ: Reducing Alert Noise
category: faq
tags: beacon,alerting,faq
---
Q: Beacon is paging our team too often for the same underlying issue. How do
we reduce this?
A: Three common options: raise the alert threshold if it's too sensitive,
add an inhibition rule so a root-cause alert suppresses related downstream
alerts, or configure a maintenance window to silence alerts during planned
work.

Q: Does silencing an alert during a maintenance window affect other teams?
A: No, maintenance windows are scoped to the specific service and alert
rule you configure them for.
