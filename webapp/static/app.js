/* NegativePluribus web table — client */
"use strict";

const $ = (id) => document.getElementById(id);
const RANKS = "23456789TJQKA";
const SUITS = ["♣", "♦", "♥", "♠"]; // c d h s
const STREETS = ["префлоп", "флоп", "терн", "ривер"];
const BB = 100;
const AUTO_NEXT_MS = 5000; // время посмотреть на результат до авто-раздачи
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let snap = null;
let bots = [];
let prevLogLen = 0;
let prevBoardLen = 0;
let busy = false;

/* ---------------- helpers ---------------- */
const chips = (n) => Math.round(n).toLocaleString("ru-RU");
const bbStr = (n) => {
  const v = n / BB;
  return (Math.abs(v - Math.round(v)) < 0.051 ? Math.round(v) : v.toFixed(1)) + "bb";
};
const sleep_ = sleep;

const rankStr = (c) => { const r = RANKS[Math.floor(c / 4)]; return r === "T" ? "10" : r; };

function cardEl(c, opts = {}) {
  const el = document.createElement("div");
  if (c === null || c === undefined) { el.className = "card back"; return el; }
  const r = rankStr(c), s = c % 4;
  el.className = "card" + (s === 1 || s === 2 ? " red" : "") + (opts.win ? " win" : "");
  if (opts.delay) el.style.animationDelay = opts.delay + "ms";
  el.innerHTML = `<div class="corner"><div class="r${r === "10" ? " ten" : ""}">${r}</div><div class="s">${SUITS[s]}</div></div>
                  <div class="pip">${SUITS[s]}</div>`;
  return el;
}

function cardsStr(cards) {
  return (cards || []).map((c) => rankStr(c) + SUITS[c % 4]).join(" ");
}

function setCards(container, cards, { win = false, delay0 = 0 } = {}) {
  container.innerHTML = "";
  if (cards) cards.forEach((c, i) => container.appendChild(cardEl(c, { win, delay: delay0 + i * 140 })));
}

async function api(path, body, timeoutMs = 30000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(path, body
      ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctl.signal }
      : { signal: ctl.signal });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("сервер не ответил за " + Math.round(timeoutMs / 1000) + " с — обнови страницу (F5)");
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3800);
}

function overlay(text) {
  $("overlay-text").textContent = text;
  $("overlay").hidden = !text;
  clearInterval(overlay._t);
  clearTimeout(overlay._heal);
  if (text) {
    const t0 = Date.now();
    overlay._t = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      const el = $("overlay-text");
      if (s > 0 && !$("overlay").hidden) el.textContent = `${text} · ${s} с`;
    }, 1000);
    overlay._heal = setTimeout(async () => {
      if ($("overlay").hidden) return;
      try {
        const st = await api("/api/state", undefined, 8000);
        if (!st.started) return;
        overlay(null);
        prevLogLen = 0;
        render(st);
        loadBots();
      } catch (e) { /* the main flow's timeout will report */ }
    }, 5000);
  }
}

/* ---------------- actions & events ---------------- */
const FRAC_LABEL = { "r0.5": "50%", "r1": "100%", "r2": "200%", "r3": "300%", "r4": "400%" };

function eventText(e) {
  switch (e.kind) {
    case "fold": return { txt: "фолд", cls: "bad" };
    case "check": return { txt: "чек", cls: "passive" };
    case "call": return { txt: `колл ${chips(e.paid)}`, cls: "" };
    case "bet": return { txt: `бет ${chips(e.paid)} (${Math.round((e.frac || 0) * 100)}%)`, cls: "aggr" };
    case "raise": return { txt: `рейз до ${chips(e.to)}`, cls: "aggr" };
    case "allin": return { txt: `олл-ин ${chips(e.to)}`, cls: "aggr" };
    default: return { txt: e.kind, cls: "" };
  }
}

function actTag(seat, e, think) {
  const el = $(seat === 0 ? "hero-act" : "villain-act");
  el.classList.remove("empty");
  if (!e && !think) { el.classList.add("empty"); el.textContent = "·"; return; }
  if (think) { el.className = "act think"; el.textContent = "думает…"; return; }
  const t = eventText(e);
  el.className = "act " + t.cls;
  el.textContent = t.txt;
}

function markActive(seat) {
  $("seat-0").classList.toggle("active", seat === 0);
  $("seat-1").classList.toggle("active", seat === 1);
}
function thinking(on) {
  $("seat-1").classList.toggle("thinking", on);
}

function setStatus() { /* removed: the turn is shown by the seat highlight itself */ }

function setBet(id, amount, who) {
  const el = $(id);
  if (!amount) { el.classList.add("empty"); el.innerHTML = ""; return; }
  el.classList.remove("empty");
  el.innerHTML = `<span class="chipstack"><i></i><i></i><i></i></span><span>${chips(amount)}</span>` +
    `<span class="who">${who} · ${bbStr(amount)}</span>`;
}

/* ---------------- rendering ---------------- */
function render(s) {
  snap = s;
  if (!s.started) { setStatus("", "выбери бота"); markActive(null); return; }

  $("hero-stack").innerHTML = `${chips(s.stacks[0])} <small>(${bbStr(s.stacks[0])})</small>`;
  $("villain-stack").innerHTML = `${chips(s.stacks[1])} <small>(${bbStr(s.stacks[1])})</small>`;
  $("hero-pos").textContent = s.heroPos;
  $("villain-pos").textContent = s.botPos;
  $("villain-name").textContent = s.bot.label;
  $("hero-dealer").hidden = !s.buttonHero;
  $("villain-dealer").hidden = s.buttonHero;

  setBet("hero-bet", s.terminal ? 0 : s.streetBets[0], "ваша ставка");
  setBet("villain-bet", s.terminal ? 0 : s.streetBets[1], "ставка бота");

  $("pot").hidden = s.pot === 0 && !s.terminal;
  $("pot-chips").textContent = chips(s.pot);
  $("pot-bb").textContent = ` · ${bbStr(s.pot)}`;
  const board = $("board");
  [...board.children].forEach((slot) => (slot.innerHTML = ""));
  s.board.forEach((c, i) => {
    if (board.children[i]) board.children[i].appendChild(cardEl(c, { delay: i >= prevBoardLen ? (i - prevBoardLen) * 170 : 0 }));
  });
  prevBoardLen = s.board.length;

  const revealAlways = $("opt-reveal").checked;
  const showBot = s.terminal && (s.terminal.showdown || revealAlways);
  setCards($("hero-cards"), s.heroHole, { delay0: 40 });
  const vc = $("villain-cards");
  vc.innerHTML = "";
  if (showBot) {
    const win = s.terminal.showdown && s.terminal.winners.includes(1);
    s.botHole.forEach((c, i) => vc.appendChild(cardEl(c, { win, delay: 40 + i * 140 })));
  } else {
    for (let i = 0; i < 2; i++) {
      const b = cardEl(null);
      b.style.animationDelay = i * 140 + "ms";
      vc.appendChild(b);
    }
  }
  if (s.terminal && s.terminal.showdown && s.terminal.winners.includes(0)) {
    setCards($("hero-cards"), s.heroHole, { win: true, delay0: 0 });
  }

  renderLog(s);
  renderHistory(s);

  const banner = $("banner");
  if (s.terminal) {
    const t = s.terminal;
    let cls = "tie", txt = "Ничья", sub = t.showdown ? "вскрытие" : "без вскрытия";
    if (t.netHero > 0) { cls = "win"; txt = `Вы выиграли +${chips(t.netHero)}`; }
    else if (t.netHero < 0) { cls = "lose"; txt = `Проигрыш −${chips(-t.netHero)}`; }
    if (t.showdown && t.winners.length === 2) txt += " (сплит)";
    banner.hidden = false;
    banner.className = "banner " + cls;
    banner.innerHTML = `${txt} <small>${sub} · ${bbStr(t.netHero)} · следующая раздача${$("opt-auto").checked ? " через несколько секунд" : " — кнопка"}</small>`;
    actTag(0, null); actTag(1, null);
    markActive(null);
    renderActions(s);
  } else {
    banner.hidden = true;
    // action tags show only the current street's actions: a new street starts clean
    const lastBy = {};
    for (const e of s.log || []) if (e.street === s.street) lastBy[e.seat] = e;
    actTag(0, lastBy[0]); actTag(1, lastBy[1]);
    markActive(s.turn === "hero" ? 0 : s.turn === "bot" ? 1 : null);
    renderActions(s);
  }

  $("s-hands").textContent = s.session.hands;
  const net = $("s-net");
  net.textContent = (s.session.netBb > 0 ? "+" : "") + s.session.netBb + " bb";
  net.style.color = s.session.netChips > 0 ? "var(--green)" : s.session.netChips < 0 ? "var(--red)" : "";
  $("s-bb100").textContent = (s.session.bb100 > 0 ? "+" : "") + s.session.bb100.toFixed(1);

  renderDecision(s.botLast, s.bot);
  $("bot-spec").textContent = s.spec || "";
}

/* stable action row: [fold][call][sizes: presets+slider][confirm] */
function renderActions(s) {
  const box = $("actions");
  box.innerHTML = "";
  if (!s.started || s.terminal) {
    const b = document.createElement("button");
    b.className = "btn primary";
    b.innerHTML = `Новая раздача <kbd>N</kbd>`;
    b.onclick = newHand;
    box.appendChild(b);
    return;
  }
  if (s.turn !== "hero") return; // bot acts fast; the seat highlight shows whose turn it is

  const callA = s.legal.find((x) => x.name === "c");
  const foldA = s.legal.find((x) => x.name === "f");
  const allinA = s.legal.find((x) => x.name === "a");
  const presets = s.legal.filter((x) => x.name.startsWith("r"));
  const bounds = s.raiseBounds;

  if (foldA) {
    const b = document.createElement("button");
    b.className = "btn fold";
    b.innerHTML = `Фолд <kbd>F</kbd>`;
    b.onclick = () => act("f");
    box.appendChild(b);
  }
  if (callA) {
    const b = document.createElement("button");
    b.className = "btn call";
    const odds = s.toCall > 0 ? ` · ${Math.round(100 * s.toCall / (s.pot + s.toCall))}%` : "";
    b.innerHTML = `<span>${callA.chips ? "Колл" : "Чек"} ${callA.chips ? chips(callA.chips) : ""}</span>` +
      (callA.chips ? `<small>${bbStr(callA.chips)}${odds}</small>` : "") + ` <kbd>C</kbd>`;
    b.onclick = () => act("c");
    box.appendChild(b);
  }

  if (bounds && (presets.length || allinA)) {
    const g = document.createElement("div");
    g.className = "sizes";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = bounds.min; slider.max = bounds.max; slider.step = 25;
    const chipsBtns = [];

    const setVal = (v, fromChip) => {
      slider.value = Math.max(bounds.min, Math.min(bounds.max, v));
      confirmBtn.innerHTML = confirmLabel(+slider.value);
      chipsBtns.forEach((c) => c.classList.toggle("on", fromChip ? c.dataset.v === String(v) : false));
    };
    const confirmLabel = (v) => {
      const isAllin = v >= bounds.max;
      const word = isAllin ? "Олл-ин" : (s.street !== "preflop" && s.toCall === 0) ? "Поставить" : "Рейз до";
      return `<span>${word} ${isAllin ? "" : chips(v) + " (" + bbStr(v) + ")"}</span><kbd>↵</kbd>`;
    };

    presets.forEach((p, i) => {
      const c = document.createElement("button");
      c.className = "chip";
      c.dataset.v = p.chips;
      c.innerHTML = `${FRAC_LABEL[p.name] || ""} <kbd>${i + 1}</kbd>`;
      c.title = `${chips(p.chips)} (${bbStr(p.chips)})`;
      c.onclick = () => setVal(p.chips, true);
      chipsBtns.push(c);
      g.appendChild(c);
    });

    slider.oninput = () => setVal(+slider.value, false);
    g.appendChild(slider);

    const confirmBtn = document.createElement("button");
    confirmBtn.className = "btn confirm";
    confirmBtn.onclick = () => act("raise", +slider.value);
    box.appendChild(g);
    box.appendChild(confirmBtn);
    renderActions._setVal = setVal;
    renderActions._slider = slider;
    setVal(presets.length ? presets[Math.min(1, presets.length - 1)].chips : bounds.min, true);
  }
}

function renderLog(s) {
  const box = $("handlog");
  box.innerHTML = "";
  let cur = null;
  for (const e of s.log || []) {
    if (e.street !== cur) {
      cur = e.street;
      const sep = document.createElement("div");
      sep.className = "street-sep";
      sep.textContent = STREETS[["preflop", "flop", "turn", "river"].indexOf(e.street)];
      if (sep.textContent) box.appendChild(sep);
    }
    const d = document.createElement("div");
    d.className = `row ${e.seat === 0 ? "hero" : "villain"}`;
    d.innerHTML = `<span class="who">${e.seat === 0 ? "вы" : "бот"}</span><span>${eventText(e).txt}</span>`;
    box.appendChild(d);
  }
  if (!s.log || !s.log.length) box.innerHTML = '<div class="muted">—</div>';
  box.scrollTop = box.scrollHeight;
}

function renderHistory(s) {
  const box = $("history");
  const items = (s.history || []).slice().reverse().slice(0, 20);
  if (!items.length) { box.innerHTML = '<div class="muted">пока пусто</div>'; return; }
  box.innerHTML = "";
  for (const h of items) {
    const d = document.createElement("details");
    const net = h.netHero;
    const cls = net > 0 ? "win" : net < 0 ? "lose" : "";
    const summ = document.createElement("summary");
    summ.innerHTML = `<span class="hn">#${h.hand}</span>` +
      `<span class="${cls}">${net > 0 ? "+" : ""}${chips(net)}</span>` +
      `<span class="hole">${cardsStr(h.heroHole)}${h.showdown ? ` / ${cardsStr(h.botHole)}` : ""}</span>` +
      `<span class="bd">${h.board.length ? "⬤ " + cardsStr(h.board) : "без борда"}</span>`;
    d.appendChild(summ);
    const body = document.createElement("div");
    body.className = "body";
    let cur = null;
    for (const e of h.log) {
      if (e.street !== cur) {
        cur = e.street;
        const sep = document.createElement("div");
        sep.className = "sep";
        sep.textContent = STREETS[["preflop", "flop", "turn", "river"].indexOf(e.street)];
        if (sep.textContent) body.appendChild(sep);
      }
      const r = document.createElement("div");
      r.className = "row";
      r.innerHTML = `<span class="who">${e.seat === 0 ? "вы" : "бот"}</span><span>${eventText(e).txt}</span>`;
      body.appendChild(r);
    }
    const brd = document.createElement("div");
    brd.className = "sep";
    brd.textContent = `борд: ${h.board.length ? cardsStr(h.board) : "—"}`;
    body.appendChild(brd);
    d.appendChild(body);
    box.appendChild(d);
  }
}

function renderDecision(last, bot) {
  const box = $("bot-decision");
  box.innerHTML = "";
  if (!last) { box.innerHTML = '<div class="muted">ещё не действовал</div>'; return; }
  if (last.fallback) {
    box.innerHTML = `<div class="fb">вне таблицы (fallback: чек/колл)</div><div class="key">${last.key}</div>`;
    return;
  }
  const items = last.legal.map((n, i) => ({ n, p: last.probs[i] })).sort((a, b) => b.p - a.p);
  for (const it of items) {
    if (it.p < 0.005) continue;
    const d = document.createElement("div");
    d.className = "prob";
    d.innerHTML = `<div class="lbl"><span>${it.n === "f" ? "фолд" : it.n === "c" ? "чек/колл" : it.n === "a" ? "олл-ин" : it.n}</span><span>${(it.p * 100).toFixed(1)}%</span></div>
                   <div class="bar"><i style="width:${Math.min(100, it.p * 100)}%"></i></div>`;
    box.appendChild(d);
  }
  const key = document.createElement("div");
  key.className = "key";
  key.textContent = last.key;
  box.appendChild(key);
  if (bot && bot.offmap && bot.offmap.decisions) {
    const fb = document.createElement("div");
    fb.className = "muted small";
    fb.textContent = `решений: ${bot.offmap.decisions}, вне таблицы: ${bot.offmap.fallbacks}`;
    box.appendChild(fb);
  }
}

/* ---------------- flow ---------------- */
async function act(name, amount) {
  if (busy || !snap || !snap.started || snap.turn !== "hero") return;
  busy = true;
  try {
    const body = { name };
    if (amount !== undefined) body.amount = amount;
    const s = await api("/api/action", body);
    render(s);
    await animateNew(s);
  } catch (e) { toast(e.message); }
  busy = false;
}

async function newHand() {
  if (busy) return;
  busy = true;
  try {
    const s = await api("/api/new", {});
    prevLogLen = 0;
    prevBoardLen = 0;
    render(s);
  } catch (e) { toast(e.message); }
  busy = false;
}

async function animateNew(s) {
  const evs = (s.log || []).slice(prevLogLen);
  prevLogLen = (s.log || []).length;
  let saidBot = false;
  let street = null;
  for (const e of evs) {
    if (e.seat === 1 && !saidBot) {
      saidBot = true;
      thinking(true);
      actTag(1, null, true);
      await sleep(750);
      thinking(false);
    }
    // a new street clears both tags: they show the current street only
    if (e.street !== street) {
      street = e.street;
      actTag(0, null); actTag(1, null);
    }
    markActive(e.seat);
    actTag(e.seat, e);
    await sleep(e.seat === 1 ? 850 : 420);
  }
  thinking(false);
  if (!s.terminal) markActive(s.turn === "hero" ? 0 : 1);
  // end state = the same rule as render: tags show the current street's actions only.
  // The replay above may have left a closing action of the PREVIOUS street on a tag.
  const lastBy = {};
  for (const e of s.log || []) if (e.street === s.street) lastBy[e.seat] = e;
  actTag(0, lastBy[0]); actTag(1, lastBy[1]);
  maybeAutoNew(s);
}

function maybeAutoNew(s) {
  clearTimeout(maybeAutoNew._t);
  if (!s.terminal || !$("opt-auto").checked) return;
  const hand = s.hand;
  maybeAutoNew._t = setTimeout(async () => {
    if (!snap || !snap.terminal || snap.hand !== hand || !$("opt-auto").checked) return;
    await newHand();
  }, AUTO_NEXT_MS);
}

async function resetSession() {
  if (busy) return;
  busy = true;
  try {
    await api("/api/reset", {});
    prevLogLen = 0;
    prevBoardLen = 0;
    await newHand();
  } catch (e) { toast(e.message); }
  busy = false;
}

async function selectBot(id) {
  overlay("грузим бота (большой файл)");
  try {
    const s = await api("/api/select", { id }, 180000);
    overlay(null);
    prevLogLen = 0;
    prevBoardLen = 0;
    render(s);
    await newHand();
    loadBots();
  } catch (e) { overlay(null); toast(e.message); }
}

async function loadBots() {
  try {
    const data = await api("/api/bots");
    bots = data.bots;
    const box = $("bot-list");
    box.innerHTML = "";
    for (const b of bots) {
      const el = document.createElement("button");
      el.className = "bot-item" + (b.active ? " active" : "");
      el.innerHTML = `<b>${b.label}</b><span>${b.note}</span>` + (b.loaded ? '<span class="tag">загружен</span>' : "");
      if (b.available === false) { el.disabled = true; el.style.opacity = "0.5"; }
      el.onclick = () => { if (!b.active && b.available !== false) selectBot(b.id); };
      box.appendChild(el);
    }
  } catch (e) { toast(e.message); }
}

/* ---------------- boot ---------------- */
/* keys by physical code: works on any keyboard layout (Russian included) */
const CODE_KEYS = { KeyF: "f", KeyC: "c", KeyA: "a", KeyN: "n", Enter: "enter" };
document.addEventListener("keydown", (ev) => {
  if (ev.repeat || !snap || !snap.started) return;
  const k = CODE_KEYS[ev.code] || (/^(Digit|Numpad)([1-9])$/.exec(ev.code) || [])[2];
  if (!k) return;
  if (k === "n" && snap.terminal) return newHand();
  if (k === "enter" && snap.turn === "hero" && !busy && renderActions._setVal) {
    return act("raise", +renderActions._slider.value);
  }
  if (snap.turn !== "hero" || busy) return;
  if (k === "f") { const a = snap.legal.find((x) => x.name === "f"); if (a) act("f"); }
  else if (k === "c") { const a = snap.legal.find((x) => x.name === "c"); if (a) act("c"); }
  else if (k === "a") { const a = snap.legal.find((x) => x.name === "a"); if (a) act("a"); }
  else if (/^[1-9]$/.test(k)) {
    const sizes = snap.legal.filter((x) => x.name.startsWith("r"));
    const a = sizes[+k - 1];
    if (a) renderActions._setVal(a.chips, true);
  }
});

$("btn-reset").onclick = resetSession;
$("opt-reveal").onchange = () => { if (snap) render(snap); };
$("opt-auto").onchange = () => {
  if (snap && !$("opt-auto").checked) clearTimeout(maybeAutoNew._t);
  if (snap && snap.terminal && $("opt-auto").checked) maybeAutoNew(snap);
};
prevLogLen = 0;
prevBoardLen = 0;
render({ started: false });
loadBots();

(async () => {
  try {
    const st = await api("/api/state");
    if (st.started) { prevLogLen = st.log ? st.log.length : 0; prevBoardLen = st.board ? st.board.length : 0; render(st); }
    else {
      const data = await api("/api/bots");
      // the default bot comes from webapp/bots.json ("default"); the server falls back to the
      // first bot whose files exist
      const def = data.bots.find((b) => b.id === data.default) || data.bots.find((b) => b.available);
      if (!def) { toast("нет ни одного бота с файлами: проверь webapp/bots.json"); return; }
      await selectBot(def.id);
    }
  } catch (e) { toast(e.message); }
})();
