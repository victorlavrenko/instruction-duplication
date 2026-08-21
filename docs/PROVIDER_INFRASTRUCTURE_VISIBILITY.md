# Provider infrastructure visibility

This note documents why AAAI reproducibility-checklist item **4.8 remains partial**.

The frozen run records the reproducible infrastructure information that was actually visible to the client: local operating system and architecture, Python implementation/version, installed package versions, exact experiment model IDs, selected backend/provider routes, route preflight results, request output ceilings, and concurrency settings. See `paper-run/config/environment.json`, `paper-run/config/models.json`, `paper-run/config/routes.json`, and `paper-run/config/preflight.json`.

The historical inference calls were served by hosted/serverless providers. Those providers did **not** expose, per request, the actual physical GPU model, CPU model, GPU/host RAM, or exact server node used. Consequently those hardware fields cannot be reconstructed truthfully after the fact. This package deliberately does not infer or invent hardware details from model names, provider marketing pages, or present-day infrastructure.

Thus 4.8 is `partial`: software/platform and hosted route identity are documented exactly, while request-level physical hardware was outside the experimenter's visibility.
