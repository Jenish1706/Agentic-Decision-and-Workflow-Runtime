---
doc_id: arch-lighthouse
title: Lighthouse Architecture Overview
category: architecture
tags: lighthouse,notifications,delivery
---
Lighthouse is Nimbus's notification delivery service, owned by team
Nimbus-Platform. It delivers messages across three channels: email, SMS, and
push notification. When a delivery attempt fails, Lighthouse retries up to 3
times using exponential backoff before giving up and moving the message to a
dead-letter queue for manual inspection.

Lighthouse is used both for on-call paging (triggered by Beacon) and for
product-facing notifications triggered by other services through its public
API.
