# Security Policy

## Supported Versions

Security fixes target the default branch until versioned releases are introduced.

## Reporting a Vulnerability

Please report vulnerabilities privately to the project maintainer before opening a public issue. Include the affected endpoint, reproduction steps, impact, and any relevant logs with secrets removed.

## Deployment Notes

- Do not commit `.env` files, API keys, customer datasets, generated reports, model artifacts, or local run logs.
- Replace `BACKEND_SESSION_SECRET` with a long random value before exposing the backend outside local development.
- Set `BACKEND_CORS_ORIGINS` to explicit frontend origins in production.
- The bundled stores are suitable for local demos and prototypes; production deployments should use durable storage, explicit retention policies, and infrastructure-level access controls.
- Rotate any credential that was ever committed before making a repository public, even if the file has since been deleted.
