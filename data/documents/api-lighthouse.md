---
doc_id: api-lighthouse
title: API Reference: Lighthouse Notify
category: api_reference
tags: lighthouse,api,notifications
---
POST /v1/notify

Sends a notification. Required fields: `channel` (one of `email`, `sms`,
`push`), `template`, and `recipient`. Optional `template_vars` object fills
placeholders in the chosen template.

Lighthouse returns a `notification_id` immediately and delivers
asynchronously. Callers that need delivery confirmation should register a
webhook URL, which Lighthouse calls with the final delivery status
(delivered, failed, or dead-lettered after 3 retries).
