/* NegativePluribus web table — client */
"use strict";

const $ = (id) => document.getElementById(id);
const RANKS = "23456789TJQKA";
const SUITS = ["♣", "♦", "♥", "♠"]; // c d h s
const STREET_RU = { preflop: "префлоп", flop: "флоп", turn: "тёрн", river: "ривер" };
const BOARD_N = { preflop: 0, flop: 3, turn: 4, river: 5 };
const BB = 100;
const AUTO_NEXT_MS = 5000; // time to look at the result before the next hand is dealt

/* ---------------- per-viewer preferences ---------------- */
const prefs = { units: "bb", speed: 1, auto: true, reveal: false };
try { Object.assign(prefs, JSON.parse(localStorage.getItem("np-prefs") || "{}")); } catch (e) { /* private mode */ }
function savePrefs() { try { localStorage.setItem("np-prefs", JSON.stringify(prefs)); } catch (e) { /* ignore */ } }

/* ---------------- state ---------------- */
let snap = null;    // the last server state
let busy = false;   // a request or a replay is running: the hero cannot act
let playId = 0;     // bumps whenever a replay must stop (new replay, bot switch, reset)
let sizeCtl = null; // the bet-size control of the current decision
let autoTimer = null;

/* ---------------- helpers ---------------- */
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const botWait = (ms) => wait(ms * prefs.speed);
const fmtChips = (n) => Math.round(n).toLocaleString("ru-RU");
const fmtBb = (n) => (n / BB).toLocaleString("ru-RU", { maximumFractionDigits: 1 }) + " bb";
const amt = (n) => (prefs.units === "bb" ? fmtBb(n) : fmtChips(n));
const amtAlt = (n) => (prefs.units === "bb" ? fmtChips(n) : fmtBb(n));
const signed = (n) => (n > 0 ? "+" : n < 0 ? "−" : "") + amt(Math.abs(n));
const rankStr = (c) => { const r = RANKS[Math.floor(c / 4)]; return r === "T" ? "10" : r; };
const isRed = (c) => c % 4 === 1 || c % 4 === 2;
const cardTxt = (c) => `<span class="${isRed(c) ? "red" : ""}">${rankStr(c)}${SUITS[c % 4]}</span>`;
const cardsTxt = (cs) => (cs || []).map(cardTxt).join(" ");
const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

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
  toast._t = setTimeout(() => (t.hidden = true), 4000);
}

function overlay(text) {
  $("overlay-text").textContent = text || "";
  $("overlay").hidden = !text;
}

/* restart a CSS animation class on an element */
function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}

/* ---------------- cards: rebuilt only when they change, so animations run once ---------------- */
function cardEl(c) {
  const el = document.createElement("div");
  if (c === null || c === undefined) { el.className = "card back"; return el; }
  const r = rankStr(c), s = c % 4;
  el.className = "card" + (isRed(c) ? " red" : "");
  el.dataset.c = c;
  el.innerHTML = `<div class="corner"><div class="r${r === "10" ? " ten" : ""}">${r}</div><div class="s">${SUITS[s]}</div></div><div class="pip">${SUITS[s]}</div>`;
  return el;
}

function setHole(el, hand, cards, hidden) {
  const key = hand + ":" + (hidden ? "back" : cards.join(","));
  if (el.dataset.key === key) return;
  const flip = !hidden && el.dataset.key === hand + ":back"; // the same hand turned face up
  el.dataset.key = key;
  el.innerHTML = "";
  cards.forEach((c, i) => {
    const e = cardEl(hidden ? null : c);
    e.classList.add(flip ? "flip" : "deal");
    e.style.animationDelay = (flip ? i * 90 : 60 + i * 130) + "ms";
    el.appendChild(e);
  });
}

function setBoard(hand, cards, n) {
  let k = 0;
  [...$("board").children].forEach((slot, i) => {
    const c = i < n ? cards[i] : null;
    const key = c === null ? "" : hand + ":" + c;
    if (slot.dataset.key === key) return;
    slot.dataset.key = key;
    slot.innerHTML = "";
    if (c !== null) {
      const e = cardEl(c);
      e.classList.add("deal");
      e.style.animationDelay = k++ * 140 + "ms";
      slot.appendChild(e);
    }
  });
}

/* showdown: the winners' five cards light up, the rest fade */
function highlight(best) {
  document.querySelectorAll("#table .card[data-c]").forEach((el) => {
    const c = +el.dataset.c;
    el.classList.toggle("best", !!best && best.has(c));
    el.classList.toggle("dim", !!best && !best.has(c));
  });
}

/* ---------------- table painting ---------------- */
const WORDS = {
  fold: ["фолд", "bad"], check: ["чек", ""], call: ["колл", ""],
  bet: ["бет", "aggr"], raise: ["рейз", "aggr"], allin: ["олл-ин", "aggr"],
};

function paintBet(id, amount, word, seat) {
  const el = $(id);
  el.classList.remove("collect-up", "collect-down");
  const key = `${amount}|${word ? word.kind : ""}`;
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  let html = "";
  if (amount > 0) html += `<span class="chipbet"><i class="chip-ico"></i>${amt(amount)}</span>`;
  if (word) {
    if (word.kind === "think") html += `<span class="word think">думает</span>`;
    else { const [t, cls] = WORDS[word.kind] || [word.kind, ""]; html += `<span class="word ${cls}">${t}</span>`; }
  }
  el.innerHTML = html;
  if (html) replay(el, "pop");
}

function paintStack(id, chips) {
  const el = $(id);
  const prev = el.dataset.v === undefined ? null : +el.dataset.v;
  el.dataset.v = chips;
  el.innerHTML = `${amt(chips)} <small>${amtAlt(chips)}</small>`;
  if (prev !== null && prev !== chips) {
    el.classList.remove("flash-up", "flash-down");
    el.classList.add(chips > prev ? "flash-up" : "flash-down");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove("flash-up", "flash-down"), 900);
  }
}

function paintPot(v) {
  const el = $("pot");
  el.classList.remove("fly-hero", "fly-bot");
  let html, cls = "pot", key;
  if (v.result) {
    const r = v.result;
    cls += " result " + r.cls;
    html = `${r.text} <small>${r.sub}</small>`;
    key = "r:" + v.hand;
  } else if (v.pot > 0) {
    html = `Банк <b>${amt(v.pot)}</b>`;
    key = "p:" + v.pot;
  } else { html = ""; key = ""; }
  if (el.dataset.key === key) return;
  const bump = el.dataset.key && el.dataset.key.startsWith("p:") && key.startsWith("p:");
  el.dataset.key = key;
  el.className = cls;
  el.innerHTML = html;
  if (bump) replay(el, "bump");
  else if (v.result) replay(el, "pop");
}

/* v: everything the table shows at one moment */
function paint(v) {
  $("hero-pos").textContent = v.heroPos;
  $("villain-pos").textContent = v.botPos;
  $("hero-pos").classList.toggle("bb", v.heroPos === "BB");
  $("villain-pos").classList.toggle("bb", v.botPos === "BB");
  paintStack("hero-stack", v.stacks[0]);
  paintStack("villain-stack", v.stacks[1]);
  setHole($("hero-cards"), v.hand, v.heroHole, false);
  setHole($("villain-cards"), v.hand, v.botHole || [null, null], !v.botHole);
  setBoard(v.hand, v.board, v.boardN);
  paintBet("hero-bet", v.bets[0], v.words[0], 0);
  paintBet("villain-bet", v.bets[1], v.thinking ? { kind: "think" } : v.words[1], 1);
  paintPot(v);
  $("hero-hand").textContent = v.heroHand || "";
  $("villain-hand").textContent = v.botHand || "";
  $("seat-0").classList.toggle("turn", v.turn === 0);
  $("seat-1").classList.toggle("turn", v.turn === 1 && !v.thinking);
  $("seat-1").classList.toggle("thinking", !!v.thinking);
  $("seat-0").classList.toggle("folded", !!v.folded[0]);
  $("seat-1").classList.toggle("folded", !!v.folded[1]);
  $("seat-0").classList.toggle("winner", !!v.winners && v.winners.includes(0));
  $("seat-1").classList.toggle("winner", !!v.winners && v.winners.includes(1));
  highlight(v.best);
}

function floatText(seat, text, cls) {
  const plaque = $(seat === 0 ? "hero-plaque" : "villain-plaque");
  const f = document.createElement("div");
  f.className = "float " + cls;
  f.textContent = text;
  plaque.appendChild(f);
  setTimeout(() => f.remove(), 1700);
}

/* action words of the given street (a new street starts clean) */
function streetWords(log, street) {
  const w = [null, null];
  for (const e of log) if (e.street === street) w[e.seat] = { kind: e.kind };
  return w;
}

/* the same combination on both sides: the pot went by the remaining cards */
const byRestCards = (t) => t && t.showdown && t.winners.length === 1 && t.heroHand && t.botHand && t.heroHand.name === t.botHand.name;

function resultOf(s) {
  const t = s.terminal;
  if (!t) return null;
  const tail = byRestCards(t) ? " · по старшим картам" : "";
  if (t.netHero > 0) return { cls: "win", text: `Вы забрали банк ${amt(t.potWon)}`, sub: signed(t.netHero) + tail };
  if (t.netHero < 0) return { cls: "lose", text: `Бот забрал банк ${amt(t.potWon)}`, sub: signed(t.netHero) + tail };
  return { cls: "tie", text: `Сплит · банк ${amt(t.potWon)}`, sub: "по нулям" };
}

/* the full, final view of a server state */
function viewOf(s) {
  const t = s.terminal;
  const showBot = !!t && (t.showdown || prefs.reveal);
  const folded = [0, 1].map((seat) => (s.log || []).some((e) => e.seat === seat && e.kind === "fold"));
  let best = null;
  if (t && t.showdown) {
    best = new Set();
    if (t.winners.includes(0) && t.heroHand) t.heroHand.best.forEach((c) => best.add(c));
    if (t.winners.includes(1) && t.botHand) t.botHand.best.forEach((c) => best.add(c));
  }
  const words = t ? [folded[0] ? { kind: "fold" } : null, folded[1] ? { kind: "fold" } : null] : streetWords(s.log, s.street);
  return {
    hand: s.hand, heroHole: s.heroHole, botHole: showBot ? s.botHole : null,
    board: s.board, boardN: s.board.length,
    bets: t ? [0, 0] : s.streetBets, stacks: s.stacks, pot: s.pot,
    heroPos: s.heroPos, botPos: s.botPos,
    words, thinking: false, folded,
    turn: t ? null : s.turn === "hero" ? 0 : 1,
    heroHand: s.heroHand ? s.heroHand.name : null,
    botHand: showBot && t.botHand ? t.botHand.name : null,
    result: resultOf(s), winners: t ? t.winners : null, best,
  };
}

/* a replay frame: the public table right after an event */
function frameView(s, f, extra) {
  return {
    hand: s.hand, heroHole: s.heroHole, botHole: null, board: s.board, boardN: f.boardN,
    bets: f.bets, stacks: f.stacks, pot: f.pot, heroPos: s.heroPos, botPos: s.botPos,
    words: [null, null], thinking: false, folded: [false, false], turn: null,
    heroHand: f.heroHand, botHand: null, result: null, winners: null, best: null, ...extra,
  };
}

/* ---------------- replay: the hand is shown event by event, never jumping ahead ---------------- */
async function collectBets(v) {
  if (!(v.bets[0] || v.bets[1])) return;
  $("hero-bet").classList.add("collect-up");
  $("villain-bet").classList.add("collect-down");
  await wait(330);
}

async function dealStreet(v, s, street, pot) {
  await collectBets(v);
  const n = BOARD_N[street];
  const added = n - v.boardN;
  const f = s.frames.find((x) => x.boardN === n);
  Object.assign(v, { bets: [0, 0], words: [null, null], pot, boardN: n, heroHand: f ? f.heroHand : v.heroHand });
  paint(v);
  await wait(260 + Math.max(0, added) * 140);
}

async function play(s, from, fresh) {
  const my = ++playId;
  const alive = () => my === playId;
  busy = true;
  snap = s;
  renderWaiting(s);
  renderPanel(s);
  const prevFolded = [0, 1].map((seat) => s.log.slice(0, from).some((e) => e.seat === seat && e.kind === "fold"));
  let f = s.frames[from];
  const v = frameView(s, f, { words: streetWords(s.log.slice(0, from), f.street), folded: prevFolded });
  let street = f.street;
  paint(v);
  if (fresh) { await wait(420); if (!alive()) return; }

  for (let i = from; i < s.log.length; i++) {
    const e = s.log[i];
    if (e.street !== street) {
      await dealStreet(v, s, e.street, s.frames[i].pot);
      if (!alive()) return;
      street = e.street;
    }
    if (e.seat === 1) {
      v.thinking = true; v.turn = 1;
      paint(v);
      await botWait(650 + Math.random() * 350);
      if (!alive()) return;
      v.thinking = false;
    }
    f = s.frames[i + 1];
    v.words = [...v.words];
    v.words[e.seat] = { kind: e.kind };
    if (e.kind === "fold") v.folded = v.folded.map((x, k) => x || k === e.seat);
    Object.assign(v, { bets: f.bets, stacks: f.stacks, pot: f.pot, turn: null, heroHand: f.heroHand });
    paint(v);
    await (e.seat === 1 ? botWait(420) : wait(200));
    if (!alive()) return;
  }

  const t = s.terminal;
  if (t) {
    // all-in run-out: the rest of the board comes street by street
    const order = ["flop", "turn", "river"];
    for (const st of order) {
      if (BOARD_N[st] > v.boardN && BOARD_N[st] <= s.board.length) {
        await dealStreet(v, s, st, s.pot);
        await botWait(450);
        if (!alive()) return;
      }
    }
    await collectBets(v);
    Object.assign(v, { bets: [0, 0], words: [null, null] });
    if (t.showdown) {
      const final = viewOf(s);
      Object.assign(v, { botHole: s.botHole, botHand: final.botHand, heroHand: final.heroHand, best: final.best, winners: t.winners });
      paint(v);
      await wait(1100);
      if (!alive()) return;
    }
    if (t.winners.length === 1) {
      $("pot").classList.add(t.winners[0] === 0 ? "fly-hero" : "fly-bot");
      await wait(460);
      if (!alive()) return;
    }
  } else if (s.street !== street) {
    await dealStreet(v, s, s.street, s.pot);
    if (!alive()) return;
  }
  busy = false;
  render(s);
  if (t) {
    if (t.netHero > 0) floatText(0, signed(t.netHero), "pos");
    else if (t.netHero < 0) floatText(1, signed(-t.netHero), "pos");
  }
}

/* ---------------- final render ---------------- */
function render(s) {
  snap = s;
  renderHeader(s);
  renderPanel(s);
  if (!s.started) { renderActions(s); return; }
  paint(viewOf(s));
  renderActions(s);
  document.title = s.turn === "hero" ? "● Ваш ход · NegativePluribus" : "NegativePluribus · стол";
  if (s.terminal) scheduleAutoNew(s);
}

function renderHeader(s) {
  const ss = s.session;
  if (!ss) return;
  $("s-hands").textContent = ss.hands;
  const net = $("s-net");
  net.textContent = (ss.netChips > 0 ? "+" : "") + ss.netBb.toLocaleString("ru-RU") + " bb";
  net.className = ss.netChips > 0 ? "pos" : ss.netChips < 0 ? "neg" : "";
  $("s-bb100").textContent = (ss.bb100 > 0 ? "+" : "") + ss.bb100.toLocaleString("ru-RU", { maximumFractionDigits: 1 });
  $("s-ci").textContent = ss.ci95 !== null && ss.ci95 !== undefined ? "± " + ss.ci95.toLocaleString("ru-RU", { maximumFractionDigits: 0 }) : "";
}

/* ---------------- action bar ---------------- */
function mkBtn(cls, html, onclick, disabled) {
  const b = document.createElement("button");
  b.className = "btn " + cls;
  b.innerHTML = html;
  if (disabled) b.disabled = true;
  if (onclick) b.onclick = onclick;
  return b;
}

/* the same row as a decision, greyed out: the layout never jumps between turns */
function renderWaiting(s, label = "Ход бота") {
  sizeCtl = null;
  const box = $("actions");
  box.innerHTML = "";
  box.className = "actions waiting";
  $("actionbar").classList.remove("my-turn");
  box.appendChild(mkBtn("fold", "Фолд", null, true));
  box.appendChild(mkBtn("call", "Чек / колл", null, true));
  const g = document.createElement("div");
  g.className = "sizes";
  box.appendChild(g);
  box.appendChild(mkBtn("confirm", "Рейз", null, true));
  const lab = document.createElement("div");
  lab.className = "wait-label";
  lab.innerHTML = `<i></i>${label}`;
  box.appendChild(lab);
  document.title = "NegativePluribus · стол";
}

function renderActions(s) {
  const box = $("actions");
  clearTimeout(autoTimer);
  sizeCtl = null;
  box.className = "actions";
  box.innerHTML = "";
  $("actionbar").classList.remove("my-turn");
  if (!s.started) {
    box.appendChild(mkBtn("primary", "Новая раздача <kbd>N</kbd>", newHand));
    return;
  }
  if (s.terminal) {
    if (s.busted) {
      box.appendChild(mkBtn("rebuy", `Сесть заново · ${amt(s.startStack)} <kbd>N</kbd>`, rebuy));
      const note = document.createElement("div");
      note.className = "bar-note";
      note.innerHTML = s.busted === "hero"
        ? `<b class="neg">Вас разорили</b> — бот забрал все фишки.`
        : `<b class="pos">Бот разорён</b> — вы забрали все фишки!`;
      box.appendChild(note);
    } else {
      const b = mkBtn("primary", `Новая раздача <kbd>N</kbd><i class="progress" id="auto-progress"></i>`, newHand);
      box.appendChild(b);
    }
    return;
  }
  if (s.turn !== "hero" || busy) { renderWaiting(s); return; }

  $("actionbar").classList.add("my-turn");
  const callA = s.legal.find((x) => x.name === "c");
  const foldA = s.legal.find((x) => x.name === "f");
  box.appendChild(mkBtn("fold", "Фолд <kbd>F</kbd>", () => act("f"), !foldA));
  if (callA) {
    const odds = s.toCall > 0 ? Math.round((100 * s.toCall) / (s.pot + s.toCall)) : 0;
    const html = callA.chips
      ? `<span class="sub">Колл ${amt(callA.chips)}<small title="шансы банка: столько эквити нужно, чтобы колл окупался">нужно ${odds}% эквити</small></span> <kbd>C</kbd>`
      : `Чек <kbd>C</kbd>`;
    box.appendChild(mkBtn("call", html, () => act("c")));
  }
  buildSizes(s, box);
}

/* bet size: presets, a slider (finer at small sizes), an exact amount field, one confirm button */
function buildSizes(s, box) {
  const bounds = s.raiseBounds;
  const g = document.createElement("div");
  g.className = "sizes";
  box.appendChild(g);
  if (!bounds) {
    box.appendChild(mkBtn("confirm", "Рейз", null, true));
    return;
  }
  const { min, max } = bounds;
  const postflop = s.street !== "preflop";
  const committed = Math.max(...s.streetBets);
  const potAfterCall = s.pot + s.toCall;
  const opts = s.legal.filter((x) => x.name.startsWith("r")).map((p) => ({
    v: p.chips,
    label: postflop ? Math.round(parseFloat(p.name.slice(1)) * 100) + "%" : amt(p.chips),
    title: postflop ? amt(p.chips) : Math.round(parseFloat(p.name.slice(1)) * 100) + "% банка",
  }));
  const allin = s.legal.find((x) => x.name === "a");
  if (allin) opts.push({ v: allin.chips, label: "олл-ин", title: amt(allin.chips) });

  const chips = opts.map((o, i) => {
    const c = document.createElement("button");
    c.className = "chip";
    c.textContent = o.label;
    c.title = `${o.title} · клавиша ${i + 1}`;
    c.onclick = () => ctl.set(o.v);
    g.appendChild(c);
    return c;
  });

  const slider = document.createElement("input");
  slider.type = "range"; slider.min = 0; slider.max = 1000; slider.step = 1;
  slider.title = "размер ставки (слева мелкие размеры крупнее)";
  g.appendChild(slider);

  const field = document.createElement("label");
  field.className = "amount";
  field.innerHTML = `<input type="text" inputmode="decimal" autocomplete="off"><span>${prefs.units === "bb" ? "bb" : "фиш."}</span>`;
  const input = field.querySelector("input");
  input.title = "точный размер: впиши и нажми Enter";
  g.appendChild(field);

  const confirm = mkBtn("confirm", "", () => act("raise", ctl.value));
  box.appendChild(confirm);

  const toPos = (v) => (max === min ? 1000 : Math.round(Math.sqrt((v - min) / (max - min)) * 1000));
  const fromPos = (p) => {
    if (p >= 1000) return max;
    const raw = min + (max - min) * (p / 1000) ** 2;
    return Math.max(min, Math.min(max, Math.round(raw / 50) * 50));
  };
  const showInput = (v) => {
    input.value = prefs.units === "bb" ? (v / BB).toLocaleString("ru-RU", { maximumFractionDigits: 2 }) : String(v);
  };
  const ctl = {
    value: min,
    set(v, src) {
      v = Math.max(min, Math.min(max, Math.round(v)));
      this.value = v;
      if (src !== "slider") slider.value = toPos(v);
      if (src !== "input") showInput(v);
      chips.forEach((c, i) => c.classList.toggle("on", opts[i].v === v));
      const isAllin = v >= max;
      const word = isAllin ? "Олл-ин" : postflop && s.toCall === 0 ? "Бет" : "Рейз до";
      const pct = Math.round((100 * (v - committed)) / potAfterCall);
      const sub = postflop ? `${pct}% банка` : amtAlt(v);
      confirm.innerHTML = `<span class="sub">${word} ${amt(v)}<small>${sub}</small></span> <kbd>↵</kbd>`;
    },
    step(dir) { this.set(this.value + dir * BB); },
    confirm() { act("raise", this.value); },
  };
  slider.oninput = () => ctl.set(fromPos(+slider.value), "slider");
  input.oninput = () => {
    const x = parseFloat(input.value.replace(",", ".").replace(/\s/g, ""));
    if (!isNaN(x)) ctl.set(prefs.units === "bb" ? x * BB : x, "input");
  };
  input.onblur = () => showInput(ctl.value);
  input.onkeydown = (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); input.blur(); ctl.confirm(); }
    ev.stopPropagation(); // typing here never fires the table shortcuts
  };
  sizeCtl = ctl;
  sizeCtl.opts = opts;
  // default: the second preset (usually ~pot), else the minimum
  ctl.set(opts.length > 1 ? opts[Math.min(1, opts.length - 2)].v : min);
}

/* ---------------- auto next hand ---------------- */
function scheduleAutoNew(s) {
  clearTimeout(autoTimer);
  if (!s.terminal || s.busted || !prefs.auto) return;
  const hand = s.hand;
  const bar = $("auto-progress");
  if (bar) { bar.style.animationDuration = AUTO_NEXT_MS + "ms"; bar.classList.add("run"); }
  autoTimer = setTimeout(() => {
    if (snap && snap.terminal && snap.hand === hand && prefs.auto && !busy) newHand();
  }, AUTO_NEXT_MS);
}

/* ---------------- flow ---------------- */
async function act(name, amount) {
  if (busy || !snap || !snap.started || snap.turn !== "hero") return;
  busy = true;
  renderWaiting(snap, "…");
  const from = snap.log.length;
  try {
    const body = { name };
    if (amount !== undefined) body.amount = amount;
    const s = await api("/api/action", body);
    await play(s, from, false);
  } catch (e) {
    busy = false;
    toast(e.message);
    if (snap) render(snap);
  }
}

async function deal(path) {
  clearTimeout(autoTimer);
  if (busy) return;
  busy = true;
  renderWaiting(snap || {}, "Раздаём…");
  try {
    const s = await api(path, {});
    await play(s, 0, true);
  } catch (e) {
    busy = false;
    toast(e.message);
    if (snap) render(snap);
  }
}
const newHand = () => deal("/api/new");
const rebuy = () => deal("/api/rebuy");

/* two clicks instead of window.confirm(): some browsers (the desktop app's pane among them) answer
   confirm() with "no" without showing it */
async function resetSession() {
  const btn = $("btn-reset");
  if (!btn.classList.contains("armed")) {
    btn.classList.add("armed");
    btn.textContent = "Точно сбросить?";
    clearTimeout(btn._t);
    btn._t = setTimeout(() => { btn.classList.remove("armed"); btn.textContent = "Сбросить сессию"; }, 3000);
    return;
  }
  clearTimeout(btn._t);
  btn.classList.remove("armed");
  btn.textContent = "Сбросить сессию";
  playId++;
  busy = false;
  clearTimeout(autoTimer);
  try {
    await api("/api/reset", {});
    await newHand();
  } catch (e) { toast(e.message); }
}

async function selectBot(id) {
  playId++;
  busy = true;
  clearTimeout(autoTimer);
  overlay("грузим бота (большой файл)…");
  try {
    const s = await api("/api/select", { id }, 180000);
    overlay(null);
    busy = false;
    render(s);
    await newHand();
  } catch (e) { overlay(null); busy = false; toast(e.message); }
  loadBots();
}

async function loadBots() {
  try {
    const data = await api("/api/bots");
    const sel = $("bot-select");
    sel.innerHTML = "";
    let active = null;
    for (const b of data.bots) {
      const o = document.createElement("option");
      o.value = b.id;
      o.textContent = b.label + (b.loaded && !b.active ? " · загружен" : "");
      o.disabled = b.available === false;
      if (b.active) { o.selected = true; active = b; }
      sel.appendChild(o);
    }
    $("bot-note").textContent = active ? active.note : "";
    return data;
  } catch (e) { toast(e.message); return null; }
}

/* ---------------- side panel ---------------- */
function renderPanel(s) {
  if (!s.session) return;
  renderChart(s.session.series || [0]);
  const m = s.session.matches || { hero: 0, bot: 0 };
  $("matches").innerHTML = m.hero || m.bot
    ? `Матчи до банкротства: вы <b class="pos">${m.hero}</b> : <b class="neg">${m.bot}</b> бот` : "";
  $("opt-carry").checked = !!s.carry;
  $("carry-note").textContent = s.carry
    ? `Бот учился на стеках ${s.startStack / BB} bb и глубину стека не различает: при других стеках он играет ту же стратегию, просто ставит не больше, чем у него есть.`
    : "Каждая раздача начинается с равных стеков — как в дуэлях.";
  if (s.bot) {
    const fb = s.bot.offmap;
    $("bot-spec").textContent = (s.spec || "") + (fb && fb.decisions ? ` · решений бота: ${fb.decisions}, вне таблицы: ${fb.fallbacks}` : "");
  }
  if (s.started) {
    $("review-title").textContent = s.terminal ? `Раздача #${s.hand} · разбор` : `Раздача #${s.hand}`;
    $("review").innerHTML = reviewHTML({
      log: s.log, board: s.board, live: !s.terminal,
      terminal: s.terminal, heroHole: s.heroHole, botHole: s.botHole,
    });
    const box = $("review");
    box.scrollTop = box.scrollHeight;
  }
  renderHistory(s);
}

const KIND_COLOR = { fold: "var(--red)", check: "var(--blue)", call: "var(--blue)", allin: "var(--orange)" };
const RAISE_COLORS = ["#f7dc9c", "#f0c46c", "#e0a445", "#c98526", "#a86d1c"];

function eventText(e) {
  switch (e.kind) {
    case "fold": return "фолд";
    case "check": return "чек";
    case "call": return `колл ${amt(e.paid)}`;
    case "bet": return `бет ${amt(e.to)} <span class="muted">(${Math.round((e.frac || 0) * 100)}%)</span>`;
    case "raise": return `рейз до ${amt(e.to)}`;
    case "allin": return `олл-ин ${amt(e.to)}`;
    default: return e.kind;
  }
}

function optLabel(o, e) {
  if (o.kind === "fold") return "фолд";
  if (o.kind === "check") return "чек";
  if (o.kind === "call") return "колл";
  if (o.kind === "allin") return "олл-ин";
  if (e.street === "preflop") return `рейз до ${amt(o.chips)}`;
  const pct = Math.round(parseFloat(o.name.slice(1)) * 100) + "%";
  return (e.toCall === 0 ? "бет " : "рейз ") + pct;
}

function mixHTML(e) {
  const p = e.policy;
  if (p.fallback) return `<div class="rv-mix"><div class="mixtext warn">этой ситуации нет в таблице бота: он играет чек/колл</div></div>`;
  let ri = 0;
  const segs = p.opts.map((o) => ({
    o, color: o.kind === "raise" ? RAISE_COLORS[Math.min(ri++, RAISE_COLORS.length - 1)] : KIND_COLOR[o.kind],
  }));
  const bar = segs.filter((x) => x.o.p >= 0.005)
    .map((x) => `<i style="flex:${x.o.p};background:${x.color}" class="${x.o.name === p.chosen ? "chosen" : ""}" title="${esc(optLabel(x.o, e))} ${(x.o.p * 100).toFixed(1)}%"></i>`).join("");
  const text = segs.filter((x) => x.o.p >= 0.01 || x.o.name === p.chosen).map((x) => {
    const lbl = `${optLabel(x.o, e)} ${Math.round(x.o.p * 100)}%`;
    return x.o.name === p.chosen ? `<b>${lbl} ✓</b>` : lbl;
  }).join(" · ");
  const chosen = p.opts.find((o) => o.name === p.chosen);
  const who = e.seat === 0 ? "blueprint на вашем месте: " : "";
  const warn = e.seat === 0 && chosen && chosen.p < 0.05 ? ` <span class="warn">— так blueprint почти не играет</span>` : "";
  return `<div class="rv-mix"><div class="mixbar">${bar}</div><div class="mixtext">${who}${text}${warn}</div></div>`;
}

function reviewHTML(h) {
  const log = h.log || [];
  if (!log.length) return '<div class="muted">—</div>';
  let out = "", cur = null;
  for (const e of log) {
    if (e.street !== cur) {
      cur = e.street;
      const n = BOARD_N[cur];
      const cards = cur === "flop" ? h.board.slice(0, 3) : n ? h.board.slice(n - 1, n) : [];
      out += `<div class="rv-street">${STREET_RU[cur]}${cards.length ? ` <span class="mini">${cardsTxt(cards)}</span>` : ""}<span class="pot-note">банк ${amt(e.potBefore)}</span></div>`;
    }
    out += `<div class="rv-row ${e.seat === 0 ? "hero" : "villain"}"><span class="who">${e.seat === 0 ? "вы" : "бот"}</span><span>${eventText(e)}</span>`;
    if (!h.live && e.policy) out += mixHTML(e);
    out += `</div>`;
  }
  const t = h.terminal;
  if (t) {
    const lines = [];
    if (h.board.length > BOARD_N[cur]) lines.push(`Доезд: ${cardsTxt(h.board.slice(BOARD_N[cur]))}`);
    if (t.showdown) {
      lines.push(`Вскрытие: вы ${cardsTxt(h.heroHole)} — ${esc(t.heroHand ? t.heroHand.name : "")}`);
      lines.push(`бот ${cardsTxt(h.botHole)} — ${esc(t.botHand ? t.botHand.name : "")}`);
      if (t.winners && byRestCards(t)) lines.push(`Комбинация одинаковая — банк решили остальные карты из лучших пяти (подсвечены на столе).`);
    } else if (prefs.reveal && h.botHole) {
      lines.push(`Карты бота: ${cardsTxt(h.botHole)}`);
    }
    lines.push(`Итог: <b class="${t.netHero > 0 ? "pos" : t.netHero < 0 ? "neg" : ""}">${signed(t.netHero)}</b>`);
    out += `<div class="rv-result">${lines.join("<br>")}</div>`;
    out += `<div class="legend"><span style="--c:var(--red)">фолд</span><span style="--c:var(--blue)">чек/колл</span><span style="--c:#f0c46c">бет/рейз</span><span style="--c:var(--orange)">олл-ин</span><span>✓ — что сыграно</span></div>`;
  } else if (h.live) {
    out += `<div class="muted small">вероятности бота покажем после раздачи — сейчас они выдали бы его руку</div>`;
  }
  return out;
}

function renderHistory(s) {
  const box = $("history");
  const items = (s.history || []).slice().reverse().slice(0, 30);
  const key = items.length ? items[0].hand + ":" + items.length + ":" + prefs.units + ":" + prefs.reveal : "";
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  if (!items.length) { box.innerHTML = '<div class="muted">пока пусто</div>'; return; }
  const open = new Set([...box.querySelectorAll("details[open]")].map((d) => d.dataset.hand));
  box.innerHTML = "";
  for (const h of items) {
    const d = document.createElement("details");
    d.dataset.hand = h.hand;
    if (open.has(String(h.hand))) d.open = true;
    const net = h.netHero;
    const name = h.showdown && h.heroHand ? h.heroHand.name : h.board.length ? "без вскрытия" : "префлоп";
    d.innerHTML = `<summary><span class="hn">#${h.hand}</span><b class="${net > 0 ? "pos" : net < 0 ? "neg" : ""}">${signed(net)}</b>` +
      `<span class="nm">${cardsTxt(h.heroHole)}${h.showdown ? " vs " + cardsTxt(h.botHole) : ""} · ${esc(name)}</span></summary>`;
    const body = document.createElement("div");
    body.className = "body review";
    const botShown = h.showdown || prefs.reveal;
    body.innerHTML = reviewHTML({
      log: h.log, board: h.board, live: false, heroHole: h.heroHole, botHole: botShown ? h.botHole : null,
      terminal: { showdown: h.showdown, netHero: net, heroHand: h.heroHand, botHand: h.botHand, winners: h.winners },
    });
    d.appendChild(body);
    box.appendChild(d);
  }
}

/* cumulative result of the session, one line; hover shows the hand */
function renderChart(series) {
  const box = $("chart");
  const key = series.length + ":" + series[series.length - 1];
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  if (series.length < 2) { box.innerHTML = '<div class="empty">график появится после первой раздачи</div>'; return; }
  const W = 320, H = 96, pad = 6;
  const lo = Math.min(0, ...series), hi = Math.max(0, ...series);
  const span = hi - lo || 1;
  const x = (i) => (i / (series.length - 1)) * W;
  const y = (v) => pad + (1 - (v - lo) / span) * (H - 2 * pad);
  const pts = series.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const last = series[series.length - 1];
  const color = last >= 0 ? "var(--green)" : "var(--red)";
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="0" x2="${W}" y1="${y(0)}" y2="${y(0)}" stroke="rgba(255,255,255,0.18)" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
      <line class="cross" x1="0" x2="0" y1="0" y2="${H}" stroke="rgba(255,255,255,0.35)" vector-effect="non-scaling-stroke" visibility="hidden"/>
    </svg>`;
  const svg = box.querySelector("svg");
  const cross = box.querySelector(".cross");
  const tip = $("tip");
  const n = (snap && snap.session && snap.session.hands) || series.length - 1;
  svg.onmousemove = (ev) => {
    const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(series.length - 1, Math.round(((ev.clientX - r.left) / r.width) * (series.length - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    const handNo = Math.round((i / (series.length - 1)) * n);
    tip.innerHTML = `после ${handNo} разд.: <b>${series[i] > 0 ? "+" : ""}${series[i].toLocaleString("ru-RU")} bb</b>`;
    tip.hidden = false;
    tip.style.left = ev.clientX + 12 + "px";
    tip.style.top = ev.clientY - 30 + "px";
  };
  svg.onmouseleave = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); };
}

/* ---------------- keyboard: physical keys, any layout ---------------- */
const CODE_KEYS = { KeyF: "f", KeyC: "c", KeyA: "a", KeyN: "n", Enter: "enter", NumpadEnter: "enter", ArrowUp: "up", ArrowDown: "down" };
document.addEventListener("keydown", (ev) => {
  if (ev.repeat || ev.ctrlKey || ev.altKey || ev.metaKey || !snap || !snap.started) return;
  if (ev.target.closest && ev.target.closest("input, select, textarea")) return;
  const k = CODE_KEYS[ev.code] || (/^(Digit|Numpad)([1-9])$/.exec(ev.code) || [])[2];
  if (!k) return;
  if (k === "n") {
    if (snap.terminal && !busy) { ev.preventDefault(); snap.busted ? rebuy() : newHand(); }
    return;
  }
  if (snap.turn !== "hero" || busy) return;
  const has = (n) => snap.legal.some((x) => x.name === n);
  if (k === "f" && has("f")) act("f");
  else if (k === "c" && has("c")) act("c");
  else if (!sizeCtl) return;
  else if (k === "a") sizeCtl.set(snap.raiseBounds.max); // only picks the size: Enter confirms
  else if (k === "enter") { ev.preventDefault(); sizeCtl.confirm(); }
  else if (k === "up" || k === "down") { ev.preventDefault(); sizeCtl.step(k === "up" ? 1 : -1); }
  else if (/^[1-9]$/.test(k)) { const o = sizeCtl.opts[+k - 1]; if (o) sizeCtl.set(o.v); }
});

/* ---------------- settings ---------------- */
function segControl(id, key, parse) {
  const seg = $(id);
  const sync = () => [...seg.children].forEach((b) => b.classList.toggle("on", parse(b.dataset.v) === prefs[key]));
  [...seg.children].forEach((b) => {
    b.onclick = () => {
      prefs[key] = parse(b.dataset.v);
      savePrefs();
      sync();
      if (snap && !busy) render(snap);
      else if (snap) renderPanel(snap);
    };
  });
  sync();
}
segControl("opt-units", "units", String);
segControl("opt-speed", "speed", Number);
$("opt-auto").checked = prefs.auto;
$("opt-reveal").checked = prefs.reveal;
$("opt-auto").onchange = () => {
  prefs.auto = $("opt-auto").checked;
  savePrefs();
  if (snap && snap.terminal && !busy) render(snap);
};
$("opt-reveal").onchange = () => {
  prefs.reveal = $("opt-reveal").checked;
  savePrefs();
  if (snap && !busy) { $("history").dataset.key = ""; render(snap); }
};
$("opt-carry").onchange = async () => {
  try {
    const s = await api("/api/settings", { carry: $("opt-carry").checked });
    snap = s;
    renderPanel(s);
    toast($("opt-carry").checked ? "Со следующей раздачи стеки переходят из раздачи в раздачу" : "Со следующей раздачи стеки снова равные");
  } catch (e) { toast(e.message); }
};
$("btn-reset").onclick = resetSession;
$("bot-select").onchange = () => selectBot($("bot-select").value);

/* ---------------- boot ---------------- */
(async () => {
  renderWaiting({}, "Загрузка…");
  try {
    const [st, data] = await Promise.all([api("/api/state"), loadBots()]);
    if (st.started) { render(st); return; }
    if (st.bot) { render(st); await newHand(); return; } // a bot is loaded, no hand yet
    if (!data) return;
    const def = data.bots.find((b) => b.id === data.default) || data.bots.find((b) => b.available);
    if (!def) { toast("нет ни одного бота с файлами: проверь webapp/bots.json"); return; }
    await selectBot(def.id);
  } catch (e) { toast(e.message); }
})();
