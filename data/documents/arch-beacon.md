---
doc_id: arch-beacon
title: Beacon Architecture Overview
category: architecture
tags: beacon,monitoring,alerting
---
Beacon is Nimbus's monitoring and alerting service, owned by team
Nimbus-Platform. It scrapes metrics from every registered Nimbus service
every 30 seconds, evaluates a set of alert rules against the scraped data,
and pages the on-call engineer through Lighthouse when a rule's threshold is
breached.

Beacon does not deliver pages itself; delivery (SMS, push notification, or
email) is handled entirely by Lighthouse, keeping the alerting-rules engine
and the notification-delivery mechanism as separate concerns.
