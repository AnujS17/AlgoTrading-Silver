"""
auth.py — Upstox OAuth2 authentication (fixed — no SSL error).

WHY THE SSL ERROR HAPPENED:
  Chrome/Edge automatically upgrades "localhost" to HTTPS (HSTS preload).
  Fix: use 127.0.0.1 (raw IP) — browsers never force HTTPS on bare IPs.
  Also replaced Flask with Python's built-in HTTPServer — zero extra deps.

HOW TO USE (run once every morning before trading):
    python auth.py

WHAT HAPPENS:
  1. Your browser opens to Upstox login
  2. You log in with your credentials + TOTP
  3. Upstox redirects to http://127.0.0.1:8765/callback
  4. This script catches it, exchanges for a token, saves to file
  5. Terminal shows "AUTHENTICATION COMPLETE"
"""

import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlencode, urlparse, parse_qs

import requests
import config

# ── Use raw IP + non-standard port to avoid HSTS/SSL issues ──
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8765

_token_received = threading.Event()
_auth_code      = None


# ─────────────────────────────────────────────────────────────
# Minimal HTTP server — just catches the one callback redirect
# ─────────────────────────────────────────────────────────────
class _CallbackHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        global _auth_code
        parsed = urlparse(self.path)

        if parsed.path != "/callback":
            self._send("<h2>Wrong path. Waiting for /callback...</h2>", 404)
            return

        params = parse_qs(parsed.query)

        if "error" in params:
            error = params["error"][0]
            self._send(f"<h2 style='color:red'>❌ Login failed: {error}</h2>", 400)
            print(f"\n❌  Upstox returned an error: {error}")
            _token_received.set()
            return

        code = params.get("code", [None])[0]
        if not code:
            self._send("<h2 style='color:red'>❌ No auth code in redirect URL</h2>", 400)
            _token_received.set()
            return

        _auth_code = code
        self._send(
            "<h2 style='color:green; font-family:sans-serif'>✅ Login successful!</h2>"
            "<p style='font-family:sans-serif'>You can close this tab and return to your terminal.</p>",
            200
        )
        _token_received.set()

    def _send(self, html: str, status: int):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass   # suppress access logs


# ─────────────────────────────────────────────────────────────
# OAuth helpers
# ─────────────────────────────────────────────────────────────
def get_auth_url() -> str:
    params = {
        "client_id":     config.API_KEY,
        "redirect_uri":  config.REDIRECT_URI,
        "response_type": "code",
    }
    return "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode(params)


def exchange_code(auth_code: str) -> dict:
    url  = "https://api.upstox.com/v2/login/authorization/token"
    data = {
        "code":          auth_code,
        "client_id":     config.API_KEY,
        "client_secret": config.API_SECRET,
        "redirect_uri":  config.REDIRECT_URI,
        "grant_type":    "authorization_code",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded",
               "Accept":       "application/json"}
    resp = requests.post(url, data=data, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed (HTTP {resp.status_code}): {resp.text}"
        )
    return resp.json()


def save_token(token_data: dict):
    token_data["saved_at"] = datetime.now().isoformat()
    with open(config.ACCESS_TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    print(f"  ✅  Token saved → {config.ACCESS_TOKEN_FILE}")


# ─────────────────────────────────────────────────────────────
# Manual fallback — user pastes the redirect URL from browser
# ─────────────────────────────────────────────────────────────
def manual_code_entry() -> str:
    print("\n" + "─"*60)
    print("  MANUAL FALLBACK")
    print("─"*60)
    print("  Your browser may show an error page after login — that's OK.")
    print("  Look at the ADDRESS BAR and copy the ENTIRE URL.")
    print("  It will look like:")
    print("  http://127.0.0.1:8765/callback?code=XXXXXXXX...\n")
    url = input("  Paste the full redirect URL here: ").strip()
    parsed = urlparse(url)
    code   = parse_qs(parsed.query).get("code", [None])[0]
    if not code:
        raise ValueError("No 'code' found in the URL you pasted.")
    return code


# ─────────────────────────────────────────────────────────────
# Used by engine.py to load token
# ─────────────────────────────────────────────────────────────
def load_token() -> str:
    if not os.path.exists(config.ACCESS_TOKEN_FILE):
        raise FileNotFoundError(
            f"\n❌  '{config.ACCESS_TOKEN_FILE}' not found.\n"
            "    Run:  python auth.py   to authenticate first.\n"
        )
    with open(config.ACCESS_TOKEN_FILE) as f:
        data = json.load(f)

    token = data.get("access_token", "")
    if not token:
        raise ValueError("Token file exists but access_token field is empty. Re-run auth.py.")

    saved_at = data.get("saved_at", "")
    if saved_at:
        saved_date = datetime.fromisoformat(saved_at).date()
        today      = datetime.now().date()
        if saved_date < today:
            raise RuntimeError(
                f"\n❌  Token saved on {saved_date}, but today is {today}.\n"
                "    Upstox tokens expire at midnight IST every day.\n"
                "    Run:  python auth.py   to get a fresh token.\n"
            )
    return token


def validate_token(token: str) -> bool:
    url  = "https://api.upstox.com/v2/user/profile"
    hdrs = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdrs, timeout=8)
        if r.status_code == 200:
            name = r.json().get("data", {}).get("user_name", "Unknown")
            print(f"  ✅  Token valid — logged in as: {name}")
            return True
        print(f"  ❌  Token check failed: HTTP {r.status_code}")
        return False
    except Exception as e:
        print(f"  ❌  Could not reach Upstox API: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Main auth flow
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  UPSTOX AUTHENTICATION — Kalman Scalper Bot")
    print("="*60)

    # ── Pre-flight: catch common config mistakes ───────────────
    if config.API_KEY in ("YOUR_API_KEY", "", None):
        print("\n❌  API_KEY is not set in config.py")
        print("   Get it from: https://developer.upstox.com → My Apps")
        sys.exit(1)

    expected = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback"
    if config.REDIRECT_URI != expected:
        print(f"\n⚠️  REDIRECT_URI mismatch:")
        print(f"   config.py has : {config.REDIRECT_URI}")
        print(f"   Should be     : {expected}")
        print(f"\n   Fix both:")
        print(f"   1. config.py  → REDIRECT_URI = \"{expected}\"")
        print(f"   2. Upstox Developer App → Redirect URL = \"{expected}\"")
        ans = input("\n   Continue anyway? (y/n): ").strip().lower()
        if ans != "y":
            sys.exit(0)

    # ── Start callback listener ────────────────────────────────
    server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _CallbackHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\n  Callback listener: http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback  ✅")

    # ── Open browser ───────────────────────────────────────────
    auth_url = get_auth_url()
    print(f"\n  Opening Upstox login in your browser...")
    print(f"\n  If the browser doesn't open, paste this URL manually:")
    print(f"  {auth_url}")
    print("\n" + "─"*60)
    print("  Waiting for you to complete login on Upstox...")
    print("  (Enter your password + TOTP/OTP on the Upstox page)")
    print("─"*60)

    threading.Timer(1.2, lambda: webbrowser.open(auth_url)).start()

    # ── Wait up to 2 minutes ───────────────────────────────────
    got_it = _token_received.wait(timeout=120)
    server.shutdown()

    if not got_it or _auth_code is None:
        print("\n⏱  Browser callback timed out. Switching to manual mode...")
        try:
            _auth_code = manual_code_entry()
        except ValueError as e:
            print(f"\n❌  {e}")
            sys.exit(1)

    # ── Exchange code → token ──────────────────────────────────
    print("\n  Exchanging auth code for access token...")
    try:
        token_data   = exchange_code(_auth_code)
        save_token(token_data)
        access_token = token_data.get("access_token", "")

        print(f"  Token: {access_token[:25]}...{access_token[-8:]}")
        print("\n  Validating with Upstox API...")
        ok = validate_token(access_token)

        print("\n" + "="*60)
        if ok:
            print("  ✅  AUTHENTICATION COMPLETE")
            print("  Next step:  python engine.py")
        else:
            print("  ⚠️  Token saved but validation failed.")
            print("  Check API_KEY and API_SECRET in config.py")
        print("="*60 + "\n")

    except RuntimeError as e:
        print(f"\n❌  {e}")
        print(f"\n  Checklist:")
        print(f"  1. API_KEY and API_SECRET correct in config.py?")
        print(f"  2. Redirect URI in Upstox app = {expected} ?")
        print(f"  3. Is your Upstox account active and verified?")
        sys.exit(1)