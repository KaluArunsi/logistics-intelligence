# Deployment Cost Policy (Startup / Zero-Budget Mode)

This policy defines hard cost constraints for development, staging, and early production.

## Goal

Run the full product with minimal spend:
- allowed paid baseline: one VPS capable of running Dockerized L0/L1 models
- everything else should be free-tier or self-hosted until paid traction exists

## Non-Negotiable Defaults

- `STRICT_ZERO_BUDGET=true`
- no mandatory paid SaaS in critical path
- no custom domain required for staging/tests
- no vendor lock-in where a free/self-hosted option exists

## Allowed in Zero-Budget Mode

- Railway deployment URL (`*.up.railway.app`) without purchased domain
- Cloudflare Worker on free tier (`*.workers.dev`)
- Self-hosted Docker services on VPS:
  - Postgres
  - Redis
  - MinIO (object storage)
  - model runner containers (L0/L1 pipelines)
- OSS/local LLM runtime where applicable

## Disallowed in Zero-Budget Mode

- paid-only managed databases as required dependency
- paid-only observability as required dependency
- paid LLM APIs as mandatory runtime path
- workflows that fail closed when paid service is unavailable

## Provider Fallback Rules

- Primary mode should be OSS/local (`LLM_PROVIDER_MODE=oss_only`).
- Paid providers may only be enabled explicitly after policy change.
- If hybrid mode is used later, fallback order should remain:
  1. local/OSS path
  2. paid path as optional fallback

## Required Runtime Flags

Use `.env.example` as baseline:
- `STRICT_ZERO_BUDGET`
- `LLM_PROVIDER_MODE`
- `ALLOW_PAID_FALLBACK`
- `ALLOW_PAID_LLM`
- `ALLOW_PAID_DATASTORES`
- `ALLOW_PAID_OBSERVABILITY`

## Deployment Profiles

- `infra` profile in `docker-compose.yml`:
  - redis, postgres, minio
- `train` profile:
  - model-runner for industry pipeline execution

## Staging / Demo Policy

- Primary staging/demo environment is Railway using platform URL only.
- Clients can be onboarded on staging URL before custom domain purchase.
- No custom domain purchase is required until:
  - validated demand
  - paying customer commitment
  - explicit budget approval

## Exit Criteria for Relaxing Policy

Policy can be relaxed only when at least one is true:
- paying client exists
- explicit budget approved for specific provider
- measurable uptime/compliance requirement cannot be met with current free/self-hosted stack

When relaxed, change log must capture:
- what paid service was approved
- reason and expected ROI
- fallback strategy if budget is removed
