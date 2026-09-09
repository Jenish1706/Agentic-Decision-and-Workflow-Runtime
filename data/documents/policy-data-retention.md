---
doc_id: policy-data-retention
title: Policy: Data Retention
category: policy
tags: data,retention,policy
---
Raw event data ingested by Aurora is retained for 30 days before automatic
deletion. Aggregated and transformed data written by Cascade to Ferry is
retained for 1 year by default, configurable per bucket.

Postmortem documents are retained indefinitely and are never subject to
automatic deletion, regardless of the age or severity of the incident they
describe.
