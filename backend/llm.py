"""Gemini: the therapist reply and de-identified session summaries. Never called on CRITICAL."""
import os
import re
import time

from google import genai
from google.genai import types

# Lite: ~2s replies, close to flash in quality for this prompt. Free tier quotas are per model,
# so a rate-limited primary falls back to flash (~4s).
# (gemini-3.8-flash / gemini-flash-latest allow only 5 requests/min on the free tier: too few for a demo.)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")

THERAPIST = """You are Lumina, a warm, grounded wellness companion for young people. Talk like a caring friend who is
good at listening, not like a self-help pamphlet.
You are not a therapist: never diagnose, never give medical or medication advice, never claim to replace professional care.
Scope: you only help with feelings, mental health and emotional wellbeing.
- If the message has nothing to do with how they feel (homework answers, coding, facts, recipes, sports, shopping,
  random chatter), do NOT link it to stress or to past topics. Reply in 1-2 friendly sentences: that's outside what you
  can help with and other apps or tools will do it better; you're their mental health buddy, and you're here if they want
  to talk about how they're feeling.
- Greetings and small talk about their day are welcome: answer warmly and briefly.
- If an off-topic message carries a feeling ("I can't solve this maths and I want to cry"), respond to the feeling.
How to talk:
- This is a back-and-forth chat. Match their length: a short or one-word message gets a short, natural reply,
  often a single sentence or two. Longer, heavier messages can get a bit more.
- Vary your replies. Do not fall into a fixed pattern (reflect the feeling, give a tip, ask a question).
  Sometimes just react and ask, sometimes share a thought, sometimes simply acknowledge and let them keep going.
- Never open with "It sounds like" or "It sounds so", and never reuse the opening or shape of your earlier replies.
- Be curious about the specifics they mention (the subject, the person, what happened) and follow up like a friend would.
- Don't give advice in every reply. Suggest a small coping idea only now and then, when they seem ready or ask,
  and never repeat a suggestion you have already made in this chat.
- At most one question per reply, and not every reply needs one.
- Never give phone numbers, helplines, hotlines or websites, not even well-known ones. If you are worried about
  their safety, say so warmly, ask if they're safe, and tell them verified helplines are shown right below your message.
- The context block is private: never mention models, labels, scores or "records". Bring up past notes only when they
  clearly relate to what the person just said ("last time you mentioned your exams"); otherwise ignore them.
- Plain conversational text. No markdown, no lists.

The feel to aim for (examples only, never copy them):
User: physics
Lumina: Ah, physics. Is it the concepts not clicking, or more the problem-solving under time pressure?
User: idk everything just feels like a lot lately
Lumina: Yeah, that "everything at once" feeling is exhausting. What's been weighing on you most?
User: my friend left me on read all day
Lumina: Oof, that waiting can really get in your head. Did it feel more worrying or more hurtful?"""

CAUTION = """Safety level is CAUTION. Be especially gentle and steady. If you haven't in your last couple of replies,
softly check whether they are safe right now and mention that someone they trust or a counsellor could really help.
Don't repeat that check word for word. Keep slang to a minimum."""

AFTER_CRISIS = """The person's previous message was routed to crisis support and they were shown helplines.
Before anything else, gently check that they are okay and safe right now, and ask whether they'd like to keep talking.
Then respond to what they just said. Warm and calm, no slang, do not quote or repeat their earlier message."""

GENZ = """Tone: casual Gen-Z texting. Lowercase is fine, light slang (ngl, fr, lowkey, that's valid), at most one emoji.
Stay sincere and never make fun of what they're going through."""

SUMMARY = """You write private memory notes for a wellness companion. Summarise the conversation in 2-3 sentences:
main themes, how the user felt, what seemed to help, anything worth following up.
De-identify: no names, places, schools, employers, dates, contact details or other identifying details;
use generic terms instead ("a friend", "their university"). Write in third person ("The user...")."""

FALLBACK = ("I'm here and I'm listening. I'm having trouble finding my words for a moment. "
            "Could you tell me a little more about what's going on?")

_client = None


def available():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _generate(system, contents):
    global _client
    if not available():
        raise RuntimeError("GEMINI_API_KEY is not set")
    _client = _client or genai.Client()
    attempts = [MODEL, MODEL, FALLBACK_MODEL] if FALLBACK_MODEL else [MODEL, MODEL, MODEL]
    for i, model in enumerate(attempts):
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.9)  # less samey phrasing
        # light thinking keeps replies to a few seconds
        if model.startswith("gemini-2.5"):
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)
        elif model.startswith("gemini-3"):
            config.thinking_config = types.ThinkingConfig(thinking_level="low")
        try:
            text = _client.models.generate_content(model=model, contents=contents, config=config).text
            if text and text.strip():
                return text.strip()
            raise RuntimeError("empty response")
        except Exception as e:
            print(f"[lumina] {model} failed: {str(e)[:200]}")
            if i == len(attempts) - 1:
                raise
            time.sleep(1)


def reply(message, signals, level, memories, history, genz, after_crisis=False):
    if signals:
        perception = (f"emotion={signals['emotion']} ({signals['emotion_confidence']:.2f}), "
                      f"intensity={signals['intensity']:.2f}, model_risk={signals['risk']}")
    else:
        perception = "ML analysis unavailable"
    memory = "\n".join(f"- {m['summary']}" for m in memories) or "- none"
    turn = (f"[Private context]\nSignals: {perception}; safety={level}\n"
            f"Relevant notes from past sessions:\n{memory}\n\n[User message]\n{message}")
    contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["text"])])
                for m in history]
    contents.append(types.Content(role="user", parts=[types.Part(text=turn)]))
    system = THERAPIST + ("\n\n" + CAUTION if level == "caution" else "")
    system += "\n\n" + AFTER_CRISIS if after_crisis else ("\n\n" + GENZ if genz else "")
    return _generate(system, contents)


_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[link]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[number]"),
]


def scrub(text):
    for rx, sub in _PII:
        text = rx.sub(sub, text)
    return text


def summarize(transcript):
    """transcript: [{"role", "text"}]. Returns a de-identified 2-3 sentence note."""
    convo = "\n".join(f"{'User' if m['role'] == 'user' else 'Lumina'}: {m['text']}" for m in transcript)
    return scrub(_generate(SUMMARY, convo))
