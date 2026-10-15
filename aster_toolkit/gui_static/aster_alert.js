/*
 * ASTER Alert card for the ASTER GUI.
 *
 * Loaded first in <head> (aster_toolkit/gui.py injects it), so it can wrap
 * WebSocket before orchestral's app.js connects to /ws. It never changes a
 * message; it only reads them:
 *
 *   - the user's outgoing {"type":"chat"} marks a live turn, so replayed
 *     history never triggers the animation;
 *   - each agent_update carries the rendered agent panel. A call to the
 *     AsterAlert tool shows as `asteralert(...)`, and its response reads
 *     "working..." until the tool returns. That drives the card:
 *     running (telescope sweeping) -> done (bell rings, headline shown).
 *
 * The tool's first output line is its headline ("ASTER Alert · ..."); the
 * card shows it. The reminder bell lists pending release-day emails from
 * GET /api/aster-alert/subscriptions.
 */
(() => {
  'use strict';

  const CALL = 'asteralert(';
  const RUNNING_COPY = {
    lookup: t => `Peeking at ${t || 'your target'}'s JWST data…`,
    watchlist: () => 'Sweeping MAST for sub-Neptunes going public…',
    check: () => 'Checking what changed since last time…',
    subscribe: t => `Setting a release-day reminder for ${t || 'you'}…`,
    unsubscribe: () => 'Removing that reminder…',
    subscriptions: () => 'Counting your reminders…',
    test_email: () => 'Sending a test email…',
  };

  // ---------------------------------------------------------------- websocket
  const NativeWS = window.WebSocket;
  let turnActive = false;
  const seen = new Map();          // "<message_index>:<nth call>" -> running|done

  function AsterWebSocket(url, protocols) {
    const ws = protocols === undefined ? new NativeWS(url) : new NativeWS(url, protocols);
    const send = ws.send.bind(ws);
    ws.send = (data) => {
      try {
        const msg = JSON.parse(data);
        if (msg && msg.type === 'chat') turnActive = true;
      } catch (_) { /* not JSON: not ours */ }
      return send(data);
    };
    ws.addEventListener('message', (ev) => {
      try { onMessage(JSON.parse(ev.data)); } catch (_) { /* never break the GUI */ }
    });
    return ws;
  }
  AsterWebSocket.prototype = NativeWS.prototype;
  for (const k of ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED']) AsterWebSocket[k] = NativeWS[k];
  window.WebSocket = AsterWebSocket;

  function onMessage(msg) {
    if (!msg || typeof msg !== 'object') return;
    if ((msg.type === 'agent_update' || msg.type === 'stream_chunk') && msg.content) {
      inspect(String(msg.content), msg.message_index);
    } else if (msg.type === 'complete' || msg.type === 'error' || msg.type === 'interrupted') {
      turnActive = false;
      if (card && card.dataset.state === 'running') {
        settle('error', 'Stopped before ASTER Alert finished.');
      }
    }
  }

  // ------------------------------------------------------------- panel parse
  const parser = new DOMParser();
  function panelText(html) {
    return parser.parseFromString(html, 'text/html').body.textContent || '';
  }

  // Rich draws boxes: strip the border glyphs and rejoin wrapped lines.
  function cleanLines(text) {
    return text.split('\n').map(l => l.replace(/^[\s│┃|]+|[\s│┃|]+$/g, ''));
  }

  function inspect(html, messageIndex) {
    if (!turnActive || html.indexOf('asteralert') === -1) return;
    const text = panelText(html);
    let at = -1, n = 0, i = text.indexOf(CALL);
    while (i !== -1) { at = i; n += 1; i = text.indexOf(CALL, i + CALL.length); }
    if (at === -1) return;

    const tail = text.slice(at);
    const r = tail.indexOf('response:');
    const callText = cleanLines(r === -1 ? tail : tail.slice(0, r)).join(' ');
    const args = {};
    for (const m of callText.matchAll(/(\w+)="([^"]*)"/g)) args[m[1]] = m[2];
    const action = (args.action || (args.target ? 'lookup' : 'watchlist')).toLowerCase();

    const response = r === -1 ? '' : tail.slice(r + 'response:'.length);
    const lines = cleanLines(response);
    while (lines.length && !lines[0]) lines.shift();
    const running = !lines.length || /^working/i.test(lines[0]);

    const key = `${messageIndex}:${n}`;
    const state = seen.get(key);
    if (running) {
      if (!state) {
        seen.set(key, 'running');
        const copy = RUNNING_COPY[action] || RUNNING_COPY.lookup;
        show('running', copy(args.target), action);
      }
      return;
    }
    if (state === 'done') return;
    seen.set(key, 'done');

    // Headline = first paragraph (wrapped lines rejoined).
    const para = [];
    for (const l of lines) { if (!l) break; para.push(l); }
    let headline = para.join(' ').replace(/\s+/g, ' ').trim();
    const failed = /^error\b/i.test(headline);
    headline = headline.replace(/^ASTER Alert\s*·\s*/, '');
    if (failed) {
      const reason = lines.find(l => /reason:/i.test(l));
      headline = reason ? reason.replace(/^-?\s*reason:\s*/i, '') : headline;
      settle('error', headline, action);
    } else {
      settle(action === 'subscribe' ? 'subscribed' : 'done', headline, action);
      refreshReminders();
    }
  }

  // ------------------------------------------------------------------- card
  const MASCOT = `
<svg class="aaf-mascot" viewBox="0 0 72 72" aria-hidden="true">
  <g class="aaf-scope">
    <rect x="30" y="9" width="24" height="9" rx="3" class="aaf-tube"/>
    <rect x="51" y="7.5" width="6" height="12" rx="2" class="aaf-lens"/>
    <circle cx="57.5" cy="13.5" r="1.6" class="aaf-glint"/>
  </g>
  <rect x="33.5" y="18" width="5" height="9" rx="2" class="aaf-mount"/>
  <circle cx="36" cy="44" r="19" class="aaf-body"/>
  <ellipse cx="29" cy="42" rx="3" ry="3.6" class="aaf-eye"/>
  <ellipse cx="43" cy="42" rx="3" ry="3.6" class="aaf-eye"/>
  <circle cx="30" cy="40.8" r="1" class="aaf-shine"/>
  <circle cx="44" cy="40.8" r="1" class="aaf-shine"/>
  <ellipse cx="24.5" cy="49" rx="3" ry="1.8" class="aaf-cheek"/>
  <ellipse cx="47.5" cy="49" rx="3" ry="1.8" class="aaf-cheek"/>
  <path d="M31.5 50 Q36 54 40.5 50" class="aaf-smile"/>
  <g class="aaf-bell">
    <path d="M58 36 q0 -7 6 -7 q6 0 6 7 v4 l2 3 h-16 l2 -3 z" class="aaf-bell-body"/>
    <circle cx="64" cy="45" r="1.8" class="aaf-bell-body"/>
  </g>
  <g class="aaf-mail">
    <rect x="54" y="30" width="16" height="11" rx="1.5" class="aaf-mail-body"/>
    <path d="M54.5 31 L62 37 L69.5 31" class="aaf-mail-flap"/>
  </g>
</svg>`;

  let card = null, hideTimer = null;

  function build() {
    if (card) return card;
    card = document.createElement('div');
    card.id = 'aster-alert-fx';
    card.className = 'aaf';
    card.setAttribute('role', 'status');
    card.setAttribute('aria-live', 'polite');
    card.innerHTML = `
      <div class="aaf-sky" aria-hidden="true">
        <i class="aaf-star" style="--x:5%;--y:20%;--d:0s"></i>
        <i class="aaf-star" style="--x:21%;--y:86%;--d:.7s"></i>
        <i class="aaf-star" style="--x:3%;--y:62%;--d:1.3s"></i>
        <i class="aaf-star" style="--x:46%;--y:7%;--d:.4s"></i>
        <i class="aaf-star" style="--x:74%;--y:6%;--d:1.8s"></i>
      </div>
      ${MASCOT}
      <div class="aaf-text">
        <div class="aaf-title">ASTER Alert</div>
        <div class="aaf-msg"></div>
        <div class="aaf-dots" aria-hidden="true"><i></i><i></i><i></i></div>
      </div>
      <button class="aaf-close" type="button" aria-label="Dismiss ASTER Alert">×</button>
      <div class="aaf-burst" aria-hidden="true">
        <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      </div>`;
    card.querySelector('.aaf-close').addEventListener('click', hide);
    card.addEventListener('mouseenter', () => clearTimeout(hideTimer));
    card.addEventListener('mouseleave', () => {
      if (card.dataset.state !== 'running') scheduleHide();
    });
    document.body.appendChild(card);
    return card;
  }

  function show(state, message, action) {
    if (!document.body) return;
    build();
    clearTimeout(hideTimer);
    card.dataset.state = state;
    card.dataset.action = action || '';
    card.querySelector('.aaf-msg').textContent = message;
    card.classList.remove('aaf-pop');
    void card.offsetWidth;            // restart the entrance animation
    card.classList.add('aaf-visible', 'aaf-pop');
    if (bell) bell.classList.add('aaf-bell-hidden');
  }

  function settle(state, message, action) {
    show(state, message || 'Done.', action || (card && card.dataset.action));
    scheduleHide();
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 12000);
  }

  function hide() {
    clearTimeout(hideTimer);
    if (card) card.classList.remove('aaf-visible');
    if (bell && bell.dataset.count !== '0') bell.classList.remove('aaf-bell-hidden');
  }

  // --------------------------------------------------------- reminder bell
  let bell = null;

  function buildBell() {
    bell = document.createElement('button');
    bell.type = 'button';
    bell.id = 'aster-alert-bell';
    bell.className = 'aaf-reminders aaf-bell-hidden';
    bell.dataset.count = '0';
    bell.innerHTML = `<span class="aaf-reminders-icon" aria-hidden="true">🔔</span>
      <span class="aaf-reminders-count"></span>
      <div class="aaf-reminders-list" role="tooltip"></div>`;
    bell.addEventListener('click', () => bell.classList.toggle('aaf-open'));
    document.body.appendChild(bell);
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  async function refreshReminders() {
    try {
      const res = await fetch('/api/aster-alert/subscriptions', { cache: 'no-store' });
      if (!res.ok) return;
      const data = await res.json();
      if (!bell) buildBell();
      const pending = data.pending || [];
      bell.dataset.count = String(pending.length);
      bell.querySelector('.aaf-reminders-count').textContent =
        pending.length === 1 ? '1 release reminder' : `${pending.length} release reminders`;
      const rows = pending.slice(0, 8).map(p =>
        `<div class="aaf-reminder"><b>${escapeHtml(p.release_date)}</b>
           ${escapeHtml(p.target)} · GO ${escapeHtml(p.proposal_id)}</div>`).join('');
      const more = pending.length > 8 ? `<div class="aaf-reminder">…and ${pending.length - 8} more</div>` : '';
      const warn = data.email_configured ? '' :
        '<div class="aaf-reminder aaf-warn">Email not configured: set ASTER_ALERT_SMTP_* in .env</div>';
      bell.querySelector('.aaf-reminders-list').innerHTML =
        `<div class="aaf-reminders-head">ASTER Alert will email you on</div>${rows}${more}${warn}`;
      const cardOpen = card && card.classList.contains('aaf-visible');
      bell.classList.toggle('aaf-bell-hidden', pending.length === 0 || cardOpen);
    } catch (_) { /* the GUI works without it */ }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', refreshReminders);
  } else {
    refreshReminders();
  }

  // For manual previews in the browser console: AsterAlertFX.demo()
  window.AsterAlertFX = {
    demo(action = 'lookup') {
      show('running', (RUNNING_COPY[action] || RUNNING_COPY.lookup)('HD-12572b'), action);
      setTimeout(() => settle(action === 'subscribe' ? 'subscribed' : 'done',
        action === 'subscribe'
          ? 'subscribed you@example.com: 1 release(s) for HD-12572b, the first on 2027-01-12'
          : 'HD-12572b: next release 2027-01-12 (in 89 days); 1 of 2 JWST time-series observations still locked',
        action), 2600);
    },
  };
})();
