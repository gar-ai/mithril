# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report vulnerabilities by:

1. **Email**: Send details to [security@example.com] (replace with actual contact)
2. **GitHub Security Advisories**: Use the "Report a vulnerability" button in the Security tab

### What to Include

Please include the following in your report:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information for follow-up

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 7 days
- **Resolution Timeline**: We aim to resolve critical vulnerabilities within 30 days
- **Disclosure**: We will coordinate with you on public disclosure timing

### Safe Harbor

We consider security research conducted in accordance with this policy to be:

- Authorized and lawful
- Exempt from DMCA claims related to circumventing security controls
- Conducted in good faith

We will not pursue legal action against researchers who:

- Make good faith efforts to avoid privacy violations and data destruction
- Do not exploit vulnerabilities beyond what's necessary to demonstrate them
- Report vulnerabilities promptly and do not disclose publicly before resolution

## Security Best Practices

When using Mithril:

1. **Keep Dependencies Updated**: Regularly update Mithril and its dependencies
2. **Validate Input**: Sanitize file paths and user inputs before processing
3. **Use Secure Storage**: Protect checkpoint files containing model weights
4. **S3 Credentials**: Use IAM roles or environment variables, never hardcode credentials
5. **Network Security**: Use TLS for remote cache operations

## Known Security Considerations

### Checkpoint Files

- Checkpoint files may contain arbitrary tensors - validate sources
- Decompressed checkpoints should be treated with the same security as original files

### S3 Storage

- Credentials are read from environment variables or AWS configuration
- No credentials are stored in cache files or logs

### Cache Directory

- Cache directories may contain compiled code artifacts
- Ensure appropriate file permissions on cache directories

## Acknowledgments

We thank the following researchers for responsibly disclosing vulnerabilities:

*(None yet - be the first!)*
