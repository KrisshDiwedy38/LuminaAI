import { Fragment, useEffect, useRef, useState } from "react";

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : `The server answered ${res.status}.`);
  }
  return res.json();
}

// Shape + colour, so levels never rely on colour alone.
const LV = {
  safe: { label: "Safe", color: "var(--safe)", bg: "var(--safe-bg)", r: "50%", rot: "0deg" },
  caution: { label: "Caution", color: "var(--caution)", bg: "var(--caution-bg)", r: "1px", rot: "45deg" },
  critical: { label: "Critical", color: "var(--critical)", bg: "var(--critical-bg)", r: "0", rot: "0deg" },
};
const STARTERS = ["Exams are stressing me out", "I can't switch my brain off at night", "I just need to vent"];
const BREATH = [["Breathe in", 4], ["Hold", 4], ["Breathe out", 6]];
const OFFLINE = "Can't reach Lumina's server. Check the backend is running, then send again.";
const REPLY_SOURCE = {
  gemini: "Written by Gemini, using the steps above.",
  crisis: "Fixed crisis text. Gemini was not called.",
  fallback: "Gemini was unavailable, so a fallback reply was used.",
  checkin: "Follows a crisis message, so Lumina first checks they're okay to continue. Written by Gemini.",
};

const btn = "min-h-10 rounded-xs px-3.5 text-sm font-semibold";
const btnOutline = `${btn} border border-line bg-surface text-ink hover:border-line-strong`;
const btnPrimary = `${btn} bg-primary text-on-primary hover:bg-primary-hover`;
const serifBody = "font-serif text-[19px] leading-[1.55] text-ink text-pretty whitespace-pre-wrap";
const small = "text-[13.5px] leading-normal text-body";

function source(a) {
  if (a.llm === "skipped") return "crisis";
  if (a.llm === "unavailable") return "fallback";
  return a.after_crisis ? "checkin" : "gemini";
}

function fmtDate(iso) {
  const d = new Date(iso);
  const part = (o) => d.toLocaleDateString("en-GB", o);
  return `${part({ weekday: "short" })}, ${d.getDate()} ${part({ month: "short" })}`;
}

function pct(x) {
  return `${Math.round(x * 100)}%`;
}

function readPref(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}

function useWidth() {
  const [w, setW] = useState(window.innerWidth);
  useEffect(() => {
    const on = () => setW(window.innerWidth);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return w;
}

export default function App() {
  const w = useWidth();
  const mode = w >= 1280 ? "desk" : w >= 1024 ? "tab" : "mob";
  const isMob = mode === "mob";

  const [status, setStatus] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState(null); // {kind: "error" | "info", text}
  const [genz, setGenz] = useState(() => readPref("lumina.genz", false));
  const [inspectorOpen, setInspectorOpen] = useState(() => window.innerWidth >= 1280);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [openSession, setOpenSession] = useState(null);
  const [selIdx, setSelIdx] = useState(null);
  const [breatheOpen, setBreatheOpen] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem("lumina.genz", JSON.stringify(genz));
    } catch {
      /* private mode: preference just isn't remembered */
    }
  }, [genz]);

  async function refreshSessions() {
    const list = await api("/sessions");
    setSessions(list);
    return list;
  }

  useEffect(() => {
    (async () => {
      try {
        setStatus(await api("/status"));
        const open = (await refreshSessions()).find((s) => !s.ended_at);
        if (open) {
          const msgs = await api(`/sessions/${open.id}/messages`);
          setSessionId(open.id);
          setMessages(msgs);
          const last = msgs.findLastIndex((m) => m.role === "assistant");
          setSelIdx(last >= 0 ? last : null);
        }
      } catch {
        setNotice({ kind: "error", text: OFFLINE });
      }
    })();
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, sending]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      if (breatheOpen) return setBreatheOpen(false);
      if (inspectorOpen && mode !== "desk") return setInspectorOpen(false);
      if (sessionsOpen) setSessionsOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [breatheOpen, inspectorOpen, sessionsOpen, mode]);

  async function send(raw) {
    const text = raw.trim();
    if (!text || sending) return;
    setNotice(null);
    setSending(true);
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setMessages((m) => [...m, { role: "user", text }]);
    try {
      let sid = sessionId;
      if (!sid) {
        sid = (await api("/sessions", { method: "POST" })).id;
        setSessionId(sid);
      }
      const res = await api(`/sessions/${sid}/messages`, { method: "POST", body: { text, genz } });
      setMessages((m) => [
        ...m.slice(0, -1),
        { role: "user", text, analysis: res.analysis },
        { role: "assistant", text: res.reply, analysis: res.analysis, resources: res.resources, support: res.support },
      ]);
      if (inspectorOpen) setSelIdx(messages.length + 1); // index of the reply just added
    } catch (e) {
      setMessages((m) => m.slice(0, -1));
      setInput(text);
      setNotice({ kind: "error", text: e.message === "Failed to fetch" ? OFFLINE : e.message });
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  async function toggleSession() {
    if (!sessionId) {
      setSessionsOpen(false);
      inputRef.current?.focus();
      return;
    }
    if (sending) return;
    try {
      const s = await api(`/sessions/${sessionId}/end`, { method: "POST" });
      setSessionId(null);
      setMessages([]);
      setSelIdx(null);
      setSessionsOpen(false);
      await refreshSessions();
      setNotice({
        kind: "info",
        text: s.summary
          ? "Session ended. Lumina kept a short private note and deleted the transcript."
          : "Session ended and the transcript was deleted.",
      });
    } catch (e) {
      setNotice({ kind: "error", text: e.message === "Failed to fetch" ? OFFLINE : e.message });
    }
  }

  function inspect(i) {
    setSelIdx(i);
    setInspectorOpen(true);
    setSessionsOpen(false);
  }

  const ended = sessions.filter((s) => s.ended_at);
  const isEmpty = messages.length === 0 && !sending;
  const canSend = input.trim().length > 0 && !sending;
  const sel = selIdx != null && messages[selIdx]?.analysis ? messages[selIdx] : null;
  const intensityPts = messages
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => m.role === "assistant" && m.analysis?.ml);
  const sbVisible = !isMob || sessionsOpen;

  return (
    <div className="relative flex h-screen overflow-hidden bg-bg text-ink">
      {sbVisible && (
        <aside
          aria-label="Past sessions"
          className={`${isMob ? "fixed inset-0 w-full" : "relative w-72"} z-40 flex h-full flex-none flex-col border-r border-line bg-bg-2`}
        >
          <div className="flex h-[60px] flex-none items-center justify-between border-b border-line px-5">
            <div className="flex items-center gap-2.5">
              <Orb size={28} />
              <span className="font-display text-lg tracking-[-0.01em]">Lumina</span>
            </div>
            {isMob && (
              <button onClick={() => setSessionsOpen(false)} className={btnOutline}>
                Close
              </button>
            )}
          </div>
          <div className="px-5 pt-5 pb-6">
            <button onClick={toggleSession} disabled={sending} className={`${btnPrimary} min-h-11 w-full disabled:opacity-60`}>
              {sessionId ? "End session" : "Start talking"}
            </button>
          </div>
          <h2 className="px-5 pb-2 text-[13px] font-semibold text-muted">Past sessions</h2>
          <div className="flex-1 overflow-y-auto px-3 pb-4">
            {ended.length ? (
              <ul>
                {ended.map((s) => {
                  const open = openSession === s.id;
                  const level = s.stats?.highest_level || "safe";
                  return (
                    <li key={s.id} className="border-b border-line">
                      <button
                        onClick={() => setOpenSession(open ? null : s.id)}
                        aria-expanded={open}
                        className="flex w-full items-center gap-3 px-2 py-3.5 text-left text-ink"
                      >
                        <Marker level={level} size={8} />
                        <span className="flex min-w-0 flex-1 flex-col gap-[3px]">
                          <span className="text-sm font-semibold">{fmtDate(s.started_at)}</span>
                          <span className="text-[12.5px] text-muted">
                            {LV[level].label} · {s.stats?.top_emotion ?? `${s.stats?.messages ?? 0} msgs`}
                          </span>
                        </span>
                        <span aria-hidden className="text-xs text-icon">
                          {open ? "▴" : "▾"}
                        </span>
                      </button>
                      {open && (
                        <p className="mr-2 mb-4 ml-7 font-serif text-[15.5px] leading-normal text-pretty text-body italic">
                          {s.summary || "No note for this session."}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="mx-2 mt-2 font-serif text-base leading-[1.55] text-pretty text-body">
                When you end a session, a short private note appears here. Lumina uses it to remember what matters.
              </p>
            )}
          </div>
          <p className="flex-none border-t border-line px-5 pt-4 pb-[18px] text-xs leading-normal text-pretty text-muted">
            Transcripts are deleted when a session ends. Only a de-identified note and mood stats are kept.
          </p>
        </aside>
      )}

      <main className="flex h-full min-w-0 flex-1 flex-col bg-bg">
        <header className="flex min-h-[60px] flex-none flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-line px-4 py-2.5">
          <div className="flex items-center gap-2">
            {isMob && (
              <button onClick={() => setSessionsOpen(true)} className={btnOutline}>
                Sessions
              </button>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <label className="flex min-h-10 cursor-pointer items-center gap-2 px-2.5 text-sm font-medium text-body">
              <input
                type="checkbox"
                checked={genz}
                onChange={(e) => setGenz(e.target.checked)}
                className="m-0 h-4 w-4 accent-[var(--primary)]"
              />
              Gen-Z tone
            </label>
            <button onClick={() => setBreatheOpen(true)} className={btnOutline}>
              Breathe
            </button>
            <button
              onClick={() => {
                setInspectorOpen(!inspectorOpen);
                setSessionsOpen(false);
              }}
              aria-pressed={inspectorOpen}
              className={inspectorOpen ? `${btn} border border-primary bg-primary text-on-primary` : btnOutline}
            >
              Under the hood
            </button>
          </div>
        </header>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto flex min-h-full max-w-[680px] flex-col px-5 pt-8 pb-7">
            {isEmpty ? (
              <div className="flex flex-1 flex-col items-center justify-center py-6 text-center">
                <Orb
                  size={isMob ? 168 : 208}
                  animate
                  glow
                  label="Start a breathing exercise"
                  onClick={() => setBreatheOpen(true)}
                />
                <h1 className="mt-11 mb-3 font-display text-[32px] leading-[1.2] font-light tracking-[-0.015em]">
                  Hey, I'm Lumina.
                </h1>
                <p className="m-0 max-w-[360px] text-base leading-relaxed text-pretty text-muted">
                  A quiet place to say what's on your mind. No judgement, no rush.
                </p>
                <div className="mt-9 flex w-full max-w-[420px] flex-col gap-2.5">
                  {STARTERS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="flex min-h-[52px] items-center justify-between gap-3 rounded-xs border border-line bg-surface-muted px-4 text-left text-[14.5px] font-semibold text-body hover:border-line-strong hover:bg-surface hover:text-primary"
                    >
                      <span>{s}</span>
                      <span aria-hidden className="text-icon">
                        →
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div aria-live="polite" className="flex flex-col gap-[30px]">
                {messages.map((m, i) => {
                  if (m.role === "user")
                    return (
                      <div key={i} className="flex justify-end">
                        <div className="max-w-[78%] rounded-[16px_16px_4px_16px] bg-primary px-4 py-3 text-[15px] leading-normal font-medium [overflow-wrap:anywhere] whitespace-pre-wrap text-on-primary">
                          {m.text}
                        </div>
                      </div>
                    );
                  const meta = m.analysis && (
                    <Meta analysis={m.analysis} selected={selIdx === i && inspectorOpen} onClick={() => inspect(i)} />
                  );
                  if (m.resources)
                    return (
                      <div key={i} className="flex flex-col gap-2.5">
                        <CrisisCard text={m.text} resources={m.resources} />
                        {meta}
                      </div>
                    );
                  return (
                    <div key={i} className="flex items-start gap-3.5">
                      <Orb size={20} style={{ marginTop: 6 }} />
                      <div className="flex min-w-0 flex-1 flex-col gap-2.5">
                        <p className={serifBody}>{m.text}</p>
                        {m.support && <SupportStrip resources={m.support} />}
                        {meta}
                      </div>
                    </div>
                  );
                })}
                {sending && (
                  <div className="flex items-center gap-3.5">
                    <Orb size={22} animate dur={3} />
                    <span aria-hidden className="flex gap-1">
                      {[0, 1, 2].map((d) => (
                        <span
                          key={d}
                          className="dot h-[5px] w-[5px] rounded-full bg-muted"
                          style={{ animationDelay: `${d * 0.2}s` }}
                        />
                      ))}
                    </span>
                    <span className="font-serif text-base text-muted italic">Lumina is writing</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex-none border-t border-line bg-bg px-4 pt-3 pb-3.5">
          <div className="mx-auto flex max-w-[680px] flex-col gap-2.5">
            {notice && (
              <div
                role={notice.kind === "error" ? "alert" : "status"}
                className="flex items-start gap-2.5 text-[13.5px] leading-[1.45]"
                style={{ color: notice.kind === "error" ? "var(--critical)" : "var(--muted)" }}
              >
                <span
                  aria-hidden
                  className="mt-1.5 h-[7px] w-[7px] flex-none"
                  style={{
                    background: notice.kind === "error" ? "var(--critical)" : "var(--muted)",
                    borderRadius: notice.kind === "error" ? 0 : "50%",
                  }}
                />
                <span className="flex-1 text-pretty">{notice.text}</span>
                <button onClick={() => setNotice(null)} className="min-h-6 flex-none px-1 text-[12.5px] text-muted underline">
                  Dismiss
                </button>
              </div>
            )}
            <div className="flex flex-col rounded-xs border border-line bg-surface focus-within:border-line-strong focus-within:shadow-[0_0_0_1px_var(--border-strong)]">
              <label htmlFor="lum-input" className="sr-only">
                Message Lumina
              </label>
              <textarea
                id="lum-input"
                ref={inputRef}
                rows={1}
                maxLength={2000}
                placeholder="Say what's on your mind…"
                value={input}
                onChange={(e) => {
                  const el = e.target;
                  el.style.height = "auto";
                  el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
                  setInput(el.value.slice(0, 2000));
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                style={{ outline: "none" }}
                className="max-h-[200px] min-h-[50px] resize-none border-0 bg-transparent px-4 pt-3.5 pb-1 text-base leading-normal text-ink placeholder:text-muted"
              />
              <div className="flex items-center justify-between gap-3 py-1.5 pr-2 pl-4">
                <span className="text-xs text-muted">
                  {input.length >= 1800
                    ? `${input.length} / 2000`
                    : isMob
                      ? "Shift+Enter for a new line"
                      : "Enter to send · Shift+Enter for a new line"}
                </span>
                <button
                  onClick={() => send(input)}
                  disabled={!canSend}
                  className={`${btn} flex-none px-5 ${canSend ? "bg-primary text-on-primary hover:bg-primary-hover" : "cursor-not-allowed bg-line text-muted"}`}
                >
                  {sending ? "Sending…" : "Send"}
                </button>
              </div>
            </div>
            <p className="m-0 text-center text-xs leading-normal text-pretty text-muted">
              Lumina is an AI companion, not a therapist or a replacement for professional care or emergency services. If
              you're in danger, call 112. For mental health support, call Tele-MANAS on 14416.
            </p>
          </div>
        </div>
      </main>

      {mode === "tab" && inspectorOpen && (
        <div onClick={() => setInspectorOpen(false)} className="fixed inset-0 z-[39] bg-scrim" />
      )}

      {inspectorOpen && (
        <aside
          aria-label="Under the hood"
          className={`${mode === "desk" ? "relative w-[380px]" : mode === "tab" ? "fixed top-0 right-0 bottom-0 w-[420px]" : "fixed inset-0 w-full"} z-[45] flex h-full flex-none flex-col border-l border-line bg-surface`}
        >
          <div className="flex h-[60px] flex-none items-center justify-between gap-3 border-b border-line pr-4 pl-6">
            <h2 className="m-0 font-display text-base font-normal tracking-[-0.01em]">Under the hood</h2>
            <button onClick={() => setInspectorOpen(false)} className={btnOutline}>
              Close
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-6">
            {sel ? (
              <Pipeline
                analysis={sel.analysis}
                userText={messages[selIdx - 1]?.text}
                points={intensityPts}
                selIdx={selIdx}
                mlTrained={status?.ml}
              />
            ) : (
              <Intro status={status} />
            )}
          </div>
        </aside>
      )}

      {breatheOpen && <Breathing onClose={() => setBreatheOpen(false)} />}
    </div>
  );
}

function Orb({ size, animate, glow, onClick, label, dur, style }) {
  const Tag = onClick ? "button" : "div";
  return (
    <Tag
      onClick={onClick}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={`orb flex-none border-0 p-0 ${animate ? "orb-anim" : ""} ${glow ? "orb-glow" : ""}`}
      style={{ width: size, height: size, ...(dur ? { animationDuration: `${dur}s` } : {}), ...style }}
    />
  );
}

function Marker({ level, size = 7 }) {
  const l = LV[level];
  return (
    <span
      aria-hidden
      className="flex-none"
      style={{ width: size, height: size, background: l.color, borderRadius: l.r, transform: `rotate(${l.rot})` }}
    />
  );
}

function Badge({ level }) {
  const l = LV[level];
  return (
    <span
      className="inline-flex items-center gap-[7px] self-start rounded-xs px-2.5 py-1 text-[12.5px] font-semibold"
      style={{ background: l.bg, color: l.color }}
    >
      <Marker level={level} />
      {l.label}
    </span>
  );
}

function Meta({ analysis, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`flex min-h-8 items-center gap-2 self-start p-0 text-[12.5px] ${selected ? "text-primary" : "text-muted"}`}
    >
      <Marker level={analysis.level} />
      <span>
        {LV[analysis.level].label}
        {analysis.ml ? `, ${analysis.ml.emotion}` : ""} ·{" "}
        <span className="underline underline-offset-[3px]">see how Lumina read this</span>
      </span>
    </button>
  );
}

function CrisisCard({ text, resources }) {
  return (
    <section
      aria-label="Reach a real person now"
      className="flex flex-col gap-5 rounded-[4px] border border-critical bg-crisis-bg px-[22px] py-6"
    >
      <Badge level="critical" />
      {text.split("\n\n").map((p, i) => (
        <p key={i} className={`m-0 ${serifBody}`}>
          {p}
        </p>
      ))}
      <ul className="border-t border-crisis-line">
        {resources.map((r) => (
          <li key={r.name} className="flex flex-col gap-1.5 border-b border-crisis-line py-3.5">
            <span className="text-sm font-semibold text-ink">{r.name}</span>
            <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 text-[15px] text-body">
              <span>Call</span>
              {r.numbers.map((n, j) => (
                <Fragment key={n}>
                  {j > 0 && <span>or</span>}
                  <a
                    href={`tel:${n.replace(/[^\d+]/g, "")}`}
                    className="font-display text-[26px] font-semibold text-ink tabular-nums underline decoration-critical decoration-2 underline-offset-[5px] hover:text-ink"
                  >
                    {n}
                  </a>
                </Fragment>
              ))}
            </span>
            {r.detail && <span className="text-sm leading-normal text-body">{r.detail}</span>}
            {r.url && (
              <a href={r.url} target="_blank" rel="noopener noreferrer" className="self-start text-sm [overflow-wrap:anywhere]">
                {r.url.replace(/^https?:\/\//, "")}
              </a>
            )}
          </li>
        ))}
      </ul>
      <p className={`m-0 ${serifBody}`}>I'm still here if you want to keep talking while you reach out.</p>
    </section>
  );
}

function SupportStrip({ resources }) {
  return (
    <aside aria-label="Verified helplines" className="flex flex-col gap-1.5 border-l-2 border-crisis-line py-1 pl-3.5">
      <span className="text-[13px] text-muted">If you want to talk to someone right now:</span>
      {resources.map((r) => (
        <span key={r.name} className="flex flex-wrap items-baseline gap-x-2 text-sm text-body">
          <span className="font-semibold text-ink">{r.name}</span>
          {r.numbers.map((n, j) => (
            <Fragment key={n}>
              {j > 0 && <span className="text-muted">or</span>}
              <a href={`tel:${n.replace(/[^\d+]/g, "")}`} className="font-semibold tabular-nums underline underline-offset-2">
                {n}
              </a>
            </Fragment>
          ))}
        </span>
      ))}
    </aside>
  );
}

function Step({ n, title, last, children }) {
  return (
    <li className="grid grid-cols-[28px_minmax(0,1fr)] gap-x-3.5">
      <div className="flex flex-col items-center">
        <span className="flex h-7 w-7 flex-none items-center justify-center rounded-full border border-line-strong text-[12.5px] font-semibold text-primary">
          {n}
        </span>
        {!last && <span className="my-1 w-px flex-1 bg-line" />}
      </div>
      <div className={`flex flex-col gap-2.5 pt-[3px] ${last ? "pb-2" : "pb-[26px]"}`}>
        <h3 className="m-0 text-sm font-semibold">{title}</h3>
        {children}
      </div>
    </li>
  );
}

function Pipeline({ analysis, userText, points, selIdx, mlTrained }) {
  const { rule, ml, level, memory } = analysis;
  const crisis = level === "critical";
  return (
    <>
      <p className="mb-2 text-[13px] text-muted">How Lumina read this message</p>
      <p className="m-0 bg-bg-2 px-3.5 py-3 text-sm leading-normal [overflow-wrap:anywhere] text-body">“{userText}”</p>
      <ol className="mt-7 flex flex-col">
        <Step n={1} title="Safety rules">
          <Badge level={rule.level} />
          <p className={`m-0 ${small}`}>
            {rule.matches.length ? `Matched: ${rule.matches.map((m) => `“${m}”`).join(", ")}` : "No risk phrases found."}
          </p>
        </Step>
        <Step n={2} title="ML perception">
          {ml ? (
            <dl className="m-0 grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-4 gap-y-2.5 text-[13.5px]">
              <dt className="text-muted">Emotion</dt>
              <dd className="m-0 text-ink">
                {ml.emotion} <span className="text-muted">· {pct(ml.emotion_confidence)}</span>
              </dd>
              <dt className="text-muted">Intensity</dt>
              <dd className="m-0 flex items-center gap-2.5">
                <span
                  role="img"
                  aria-label={`Intensity ${ml.intensity.toFixed(2)} of 1`}
                  className="block h-1.5 flex-1 border border-line bg-bg-2"
                >
                  <span className="block h-full bg-primary" style={{ width: pct(ml.intensity) }} />
                </span>
                <span className="text-ink tabular-nums">{ml.intensity.toFixed(2)}</span>
              </dd>
              <dt className="text-muted">Risk</dt>
              <dd className="m-0 text-ink">
                {LV[ml.risk].label} <span className="text-muted">· {pct(ml.risk_confidence)}</span>
              </dd>
            </dl>
          ) : (
            <p className={`m-0 ${small}`}>Model not trained yet. Safety rules decide this message as is.</p>
          )}
        </Step>
        <Step n={3} title="Safety decision">
          <Badge level={level} />
          <p className={`m-0 text-pretty ${small}`}>
            {crisis
              ? "Critical goes straight to the crisis path. The AI is skipped and fixed helpline text is shown."
              : "The stricter of the rules and the model decides. The model can raise the level, never lower it."}
          </p>
        </Step>
        <Step n={4} title="Memory">
          {memory.length ? (
            memory.map((m, i) => (
              <figure key={i} className="m-0 bg-bg-2 px-3.5 py-3">
                <blockquote className="m-0 font-serif text-[15.5px] leading-normal text-ink italic">“{m.summary}”</blockquote>
                <figcaption className="mt-2 text-xs text-muted tabular-nums">
                  {fmtDate(m.date)} · similarity {(1 - m.distance).toFixed(2)}
                </figcaption>
              </figure>
            ))
          ) : (
            <p className={`m-0 ${small}`}>{crisis ? "Skipped on the crisis path." : "Nothing relevant from past sessions."}</p>
          )}
        </Step>
        <Step n={5} title="Reply" last>
          <p className={`m-0 text-pretty ${small}`}>{REPLY_SOURCE[source(analysis)]}</p>
          {analysis.stripped && (
            <p className={`m-0 text-pretty ${small}`}>
              Removed a phone number or website Gemini wrote itself. Only verified helplines are shown.
            </p>
          )}
          {analysis.support && (
            <p className={`m-0 text-pretty ${small}`}>Verified helplines were attached below the reply.</p>
          )}
        </Step>
      </ol>
      <div className="mt-7 border-t border-line pt-5">
        <h3 className="mt-0 mb-3.5 text-sm font-semibold">Intensity this session</h3>
        {mlTrained && points.length >= 2 ? (
          <>
            <div
              role="img"
              aria-label={`Intensity per reply: ${points.map(({ m }) => m.analysis.ml.intensity.toFixed(2)).join(", ")}`}
              className="flex h-[72px] items-end gap-1.5 border-b border-line"
            >
              {points.map(({ m, i }) => (
                <span
                  key={i}
                  className="max-w-7 flex-1"
                  style={{
                    height: Math.round(m.analysis.ml.intensity * 60 + 6),
                    background: LV[m.analysis.level].color,
                    opacity: i === selIdx ? 1 : 0.55,
                  }}
                />
              ))}
            </div>
            <div className="mt-1.5 flex justify-between text-xs text-muted">
              <span>First reply</span>
              <span>Latest</span>
            </div>
          </>
        ) : (
          <p className="m-0 text-[13.5px] text-muted">
            {mlTrained ? "Needs at least two replies to draw." : "Available once the ML model is trained."}
          </p>
        )}
      </div>
    </>
  );
}

function Intro({ status }) {
  const rows = [
    {
      label: "ML model:",
      ok: status?.ml,
      text: !status ? "unknown (server offline)" : status.ml ? `loaded (${status.ml_config?.base_model ?? "custom"})` : "not trained yet",
    },
    {
      label: "Gemini:",
      ok: status?.llm,
      text: !status ? "unknown (server offline)" : status.llm ? `ready (${status.model})` : "unavailable (no API key)",
    },
  ];
  return (
    <>
      <p className="m-0 font-serif text-[17px] leading-[1.55] text-pretty text-ink">
        Every reply passes through five steps: safety rules, an ML read of emotion and risk, a safety decision, memory of
        past sessions, then the reply itself.
      </p>
      <p className="mt-3.5 mb-0 text-sm leading-normal text-pretty text-muted">
        Choose “see how Lumina read this” under any reply to walk through it.
      </p>
      <h3 className="mt-7 mb-3 text-sm font-semibold">System status</h3>
      <dl className="m-0 flex flex-col gap-2.5 text-[13.5px]">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2.5">
            <span
              aria-hidden
              className="h-[7px] w-[7px] rounded-full"
              style={{ background: r.ok ? "var(--safe)" : "var(--icon)" }}
            />
            <dt className="text-muted">{r.label}</dt>
            <dd className="m-0 text-ink">{r.text}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

function Breathing({ onClose }) {
  const [started, setStarted] = useState(false);
  const [phase, setPhase] = useState(0);
  const [left, setLeft] = useState(BREATH[0][1]);
  const [rounds, setRounds] = useState(0);
  const doneRef = useRef(null);

  useEffect(() => {
    doneRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!started) return;
    const t = setTimeout(() => {
      if (left > 1) return setLeft(left - 1);
      const next = (phase + 1) % BREATH.length;
      if (phase === BREATH.length - 1) setRounds((r) => r + 1);
      setPhase(next);
      setLeft(BREATH[next][1]);
    }, 1000);
    return () => clearTimeout(t);
  }, [started, phase, left]);

  const scale = !started || phase === 2 ? 0.62 : 1;
  const dur = !started ? 0.6 : phase === 0 ? 4 : phase === 2 ? 6 : 0.2;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="lum-breath-title"
      className="fixed inset-0 z-[70] flex flex-col bg-bg"
    >
      <div className="flex justify-end px-5 py-4">
        <button ref={doneRef} onClick={onClose} className={`${btnOutline} min-h-11 px-5`}>
          I'm done
        </button>
      </div>
      <div className="flex flex-1 flex-col items-center justify-center p-6 text-center">
        <div className="flex h-[min(300px,72vw)] w-[min(300px,72vw)] items-center justify-center">
          <Orb
            size="100%"
            glow
            style={{ transform: `scale(${scale})`, transition: `transform ${dur}s cubic-bezier(.45,0,.55,1)` }}
          />
        </div>
        <h2
          id="lum-breath-title"
          aria-live="polite"
          className="mt-11 mb-2 font-display text-[30px] font-light tracking-[-0.015em]"
        >
          {started ? BREATH[phase][0] : "Get comfortable"}
        </h2>
        <p className="m-0 min-h-[26px] text-[17px] text-muted tabular-nums">
          {started ? left : "In for 4, hold for 4, out for 6"}
        </p>
        <p className="mt-1.5 mb-0 min-h-[22px] text-sm text-muted">
          {rounds ? `${rounds} round${rounds > 1 ? "s" : ""} done` : ""}
        </p>
        {!started && (
          <button onClick={() => setStarted(true)} className={`${btnPrimary} mt-6 min-h-11 px-7`}>
            Start
          </button>
        )}
      </div>
    </div>
  );
}
