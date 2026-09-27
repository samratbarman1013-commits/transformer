# Privacy promise

**We store nothing.** Specifically:

1. **Chat history lives only on your device.** The web client keeps
   conversations in the browser's localStorage. Nothing about your
   conversations is written to any server database — the server has no
   database at all.

2. **The server is stateless per request.** Each request is processed in
   memory and discarded. There are no accounts, no sessions, no analytics.

3. **Memory (preferences) is local.** The agent's memory tool writes to a JSON
   file on the *server operator's* machine only when the *operator* runs it
   locally — in the hosted setup this tool is for self-hosters, and the hosted
   demo does not enable persistent memory.

4. **Export/import.** You can export everything the app knows about you as
   JSON and delete it from the device at any time (clearing site data removes
   it permanently).

## Caveats we will not hide

- Debugging logs (Phase 3) are **structured run logs** (tool call name, timing,
  error status) intended for the server operator's local debugging. The
  default policy is to log metadata, not message content; operators who change
  that are responsible for disclosing it.
- If web search is enabled, your query necessarily goes to the search
  provider, under the server operator's API key.
- "We store nothing" describes *our* hosted deployment and the default
  configuration. Anyone can fork the project and change it — that's what
  open source means. Audit the code; it's short.
