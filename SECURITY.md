# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✅ |
| < Latest | ❌ |

We support only the most recent release. Please update before reporting a vulnerability.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately by emailing **mboakes@turing.ac.uk**. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The version(s) of Jabberjay affected
- Any suggested mitigations, if you have them

You can expect an acknowledgement within **48 hours** and a status update within **7 days**.

## Scope

Jabberjay is a model inference library. Security concerns most likely to be relevant:

- **Arbitrary code execution** via malicious model weights or YAML config files
- **Path traversal** in audio file loading
- **Dependency vulnerabilities** in `torch`, `transformers`, or `librosa`

Findings in third-party dependencies should be reported upstream to those projects as well.
