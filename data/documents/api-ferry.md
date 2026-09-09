---
doc_id: api-ferry
title: API Reference: Ferry Storage
category: api_reference
tags: ferry,api,storage
---
PUT /v1/objects/{bucket}/{key} — uploads an object.
GET /v1/objects/{bucket}/{key} — downloads an object.
DELETE /v1/objects/{bucket}/{key} — deletes an object (subject to the
storage cleanup policy for objects older than 30 days).

GET /v1/objects/{bucket}/{key}?presign=true returns a time-limited presigned
URL instead of the object body, useful for letting a client download
directly without proxying through the caller.

Adding `?version={version_id}` to any read request retrieves a specific past
version of an object, if versioning is enabled on the bucket.
