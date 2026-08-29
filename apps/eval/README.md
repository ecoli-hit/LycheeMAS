# LycheeMAS Eval clients

This directory contains independently runnable interfaces to the same Eval
application services:

- `server/`: Python composition root for the HTTP API and scheduler lifecycle.
- `web/`: React Web client branded as LycheeMAS Eval Studio.
- `tui/`: contract placeholder for a future terminal UI client.
- `cli/`: contract placeholder for a future user-facing management CLI.

The TUI and CLI currently contain documentation only. Their first executable
slice should call the versioned HTTP API through a shared Python client rather
than read registries or run artifacts directly.
