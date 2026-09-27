# Android app

## Where we are

The web client (`app/web/`) is a **PWA**: installable on Android straight from
the browser (Chrome → "Install app"), with an app icon, standalone window,
offline notice, and on-device history. That ships today with zero extra
code — one codebase serves web + installable Android.

## The APK (Phase 5)

When we wrap it, the plan is a thin **WebView shell**:

- `MainActivity` with a `WebView` pointed at the deployed site (or a bundled
  copy of `app/web/` for fully-offline install)
- Back-button and file-download handling (for JSON export)
- Same privacy model: all state on-device; the APK talks to the API only

A Flutter port is possible later, but the WebView shell keeps the APK at a few
hundred kilobytes and keeps one UI codebase. That is the default unless we need
native features (notifications, share targets).

## Why not build the APK right now

Building a signed APK needs the Android SDK + Gradle toolchain. The PWA gives
the same installable experience immediately; the WebView shell is a few hours
of work when the model is actually serving responses worth installing an app
for.
