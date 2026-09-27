# SynTrail-LM Issue #7: close during generation

## Goal

Preserve the dirty-model close confirmation when the user requests window
closure while Chat generation owns the `AppHandle`.

## Scope

- Observe native viewport close requests in the GUI update loop.
- Cancel and remember close requests while generation is in flight.
- After the worker returns the handle, reuse the normal discard confirmation and
  close only when it allows the operation.
- Keep the worker/panic recovery design unchanged.
- Leave the separate P3 crash-state and source-assertion cleanup for a follow-up
  issue.

## Verification

- [x] Add a source-wiring regression test and observe the expected RED failure.
- [x] Add the close-request decision helper and deferred-close state machine.
- [x] Run `cargo test --test chat_gui_responsiveness`.
- [x] Run `cargo test --features gui --bin syntrail-gui`.
- [x] Run the complete GUI-enabled test suite on the exact commit.
- [ ] Review the exact commit and verify CI before merge.
