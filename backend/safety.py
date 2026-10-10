"""Deterministic safety gate. Runs on every message, independent of the ML model.

Phrases are written plainly and compiled to tolerant regexes:
  - each letter may repeat ("kiiiill" matches "kill", "kil" also matches)
  - word gaps are optional ("killmyself", "kill-myself")
  - "(a|b)" groups and "?" work as in regex
Text is normalised first: NFKC, lowercase, leetspeak (k1ll, $uicide), punctuation -> space.
Every change here needs a case in tests/test_safety_rules.py.
"""
import difflib
import re
import unicodedata

SAFE, CAUTION, CRITICAL = "safe", "caution", "critical"
SEVERITY = {SAFE: 0, CAUTION: 1, CRITICAL: 2}

_I = "(i|im|i am|ive|i have|i will|ill|i would|id|i)? ?"
# no "going to"/"gonna": "im gonna die if i fail this exam" is everyday hyperbole
_WANT = "(want|wanna|wana|need|plan|planning|ready|trying|try|tried)( to)?"

CRITICAL_PHRASES = [
    # suicide / self-harm intent
    "kill (myself|my self|meself)", "kms",
    "(end|take|ending|taking) my (own )?life", "end it all", "ending it all",
    f"{_I}{_WANT} (die|be dead)", "wish i (was|were) dead", "better off dead",
    f"{_I}{_WANT} end it", "(going to|gonna) end (it all|my life)","dont want to (live|be alive|exist|wake up)",
    "dont wanna (live|be alive|exist|wake up)",
    "dont want to be here anymore", "dont wanna be here anymore",
    "no reason to (live|be alive|keep going)", "nothing to live for",
    "(everyone|they|world|family|friends) (would be|is|are|will be) better (off )?without me",
    "unalive", "unaliving", "sewerslide", "suicide", "suicidal", "commit sui",
    "(hang|hanging|drown|drowning|shoot|shooting|stab|stabbing|starve|starving) myself",
    "(cut|cutting|hurt|hurting|harm|harming|burn|burning) myself",
    "self ?harm", "slit my (wrist|wrists|throat)",
    "(take|took|taking|swallow|swallowed) (all|a bunch of|a whole bottle of|too many) (the |my |of )?pills",
    "overdose", "od on",
    "(jump|jumping) (off|from) (a|the|this) (bridge|building|roof|cliff)",
    "(jump|step|jumping|stepping) in front of (a|the) (train|car|bus|truck)",
    "say goodbye to everyone", "(wrote|writing) (a|my) (suicide |goodbye )?note",
    # asking for a method
    "(easiest|easy|painless|quickest|quick|fastest|best|simplest) way to (die|kill myself|end it|end my life)",
    "how (to|do i|can i|could i|should i) (die|kill myself|end it|end my life)",
    "how many pills",
]

CAUTION_PHRASES = [
    "hopeless", "worthless", "(i|im|i am|i feel like|i feel)( such| just| so)?( a)? (burden|failure|useless)", "hate myself", "hate my life",
    "cant (do|take) (this|it) anymore", "cant go on", "give up on (everything|life)", "giving up on (everything|life)",
    "whats the point", "no point (in )?(anything|living|trying)", "nobody (cares|would care|would notice)",
    "no one (cares|would care|would notice)", "(want|wanna) (to )?disappear", "tired of (living|everything|existing)",
    "(feel|feeling) empty", "numb inside", "(panic|anxiety) attack", "cant breathe",
    "(havent|cant|not) (slept|eaten|sleeping|eating)", "trapped", "(used to|stopped) (cutting|self harming)",
    "(he|she|they|dad|mom|mum|father|mother|parents|brother|sister|boyfriend|girlfriend|partner|bf|gf)"
    " (hits|hit|hurts|hurt|abuses|abused|beats|beat) me", "not safe at home",
    # harm to others: often hyperbole ("i want to kill him lol"), so CAUTION and the reply checks in
    f"{_I}{_WANT} (kill|hurt|shoot|stab) (him|her|them|someone|somebody|people|everyone)",
    # curiosity about dying: not intent, but worth a gentle check-in
    "how (does|would|do|will) (dying|death|it) feel( like)?( to die)?",
    "what (does|is|would) (dying|death|it) (feel )?like( to die)?", "what happens (when|after) (you|i|we|people|u) die",
    "(thinking|think|thought|thoughts) (about|of) (death|dying|being dead)",
]

# A vague wish ("i wanna feel that") is only as safe as what "that" refers to. Within the last few
# user messages, talk of death or dying turns it into CRITICAL.
FOLLOW_UP_PHRASES = [
    f"{_I}({_WANT}|wish|would like|d like|wanna)( to)? (feel|do|try|experience|have|go through) (that|it|this)",
    f"{_I}(want|wanna|wish for|need) (that|it|this)( too)?",
    "(that|it|this) (sounds|would be|seems|feels|must be|must feel) (so )?(nice|peaceful|good|better|nicer|easier|like a relief|freeing)",
    "(i )?wish (that|it) (was|were) me",
]
DEATH_TOPIC_WORDS = ["die", "died", "dying", "death", "dead", "suicide", "suicidal", "kill myself", "end my life",
                     "end it", "overdose", "not wake up"]
RECENT_MESSAGES = 3  # how far back a follow-up looks

# Fuzzy single-word misspellings ("sucide", "suicdal"). Only words this long are checked.
FUZZY_CRITICAL = ["suicide", "suicidal", "overdose"]
FUZZY_MIN_LEN, FUZZY_RATIO = 6, 0.85  # 0.8 lets "overdue" match "overdose"

# Explicit denials are downgraded to CAUTION, never to SAFE ("im not suicidal but..." still checks the rest).
NEGATED_PHRASES = [
    "(not|never|wasnt|isnt|arent|wouldnt|would never|will never|dont) (be )?(feeling )?suicidal",
    "(would|will) never (kill|hurt|harm) myself", "not (going|gonna|trying) to (kill|hurt|harm) myself",
    "not (thinking|thought) (of|about) (suicide|killing myself)",
    # "i dont wanna die" contains "wanna die"; only die / be dead / self-harm objects, so
    # "i dont want to live" is untouched and stays CRITICAL
    "(dont|do not|didnt|did not|never) (want|wanna|wana|plan|mean|need)( to)? (die|be dead|kill myself|hurt myself|end my life)",
]

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _compile(phrase):
    out, prev = [], None
    for c in phrase:
        if c.isalnum():
            if c != prev:
                out.append(re.escape(c) + "+")
            prev = c
            continue
        prev = None
        out.append({" ": r"\s*", "(": "(?:"}.get(c, c))
    return re.compile(r"\b" + "".join(out) + r"\b")


_CRIT = [(p, _compile(p)) for p in CRITICAL_PHRASES]
_CAUT = [(p, _compile(p)) for p in CAUTION_PHRASES]
_NEG = [_compile(p) for p in NEGATED_PHRASES]
_FOLLOW = [_compile(p) for p in FOLLOW_UP_PHRASES]
_DEATH = [_compile(p) for p in DEATH_TOPIC_WORDS]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("'", "").replace("’", "")
    # leetspeak only inside words that contain a letter, so "2 days" or "100%" are left alone
    text = re.sub(r"\S+", lambda m: m.group().translate(_LEET) if re.search(r"[a-z]", m.group()) else m.group(), text)
    text = re.sub(r"(?<=[a-z])[!|](?=[a-z])", "i", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check(text, recent=()):
    """Return {"level": safe|caution|critical, "matches": [phrases that fired]}.
    recent: the user's previous messages in this session, newest first."""
    norm = normalize(text)
    matches, level = [], SAFE

    negated = [m.group() for rx in _NEG for m in rx.finditer(norm)]
    for rx in _NEG:
        norm = rx.sub(" ", norm)
    if negated:
        level = CAUTION
        matches += [f"denial: {n}" for n in negated]

    crit = [m.group().strip() for _, rx in _CRIT if (m := rx.search(norm))]
    crit += [f"{w} (spelled like {k})" for w in norm.split() if len(w) >= FUZZY_MIN_LEN
             for k in FUZZY_CRITICAL if difflib.SequenceMatcher(None, w, k).ratio() >= FUZZY_RATIO]
    follow = next((m.group().strip() for rx in _FOLLOW if (m := rx.search(norm))), None)
    if not crit and follow:
        topic = next((r for r in list(recent)[:RECENT_MESSAGES]
                      if any(rx.search(normalize(r)) for rx in _DEATH)), None)
        if topic:
            crit = [f"{follow} (right after: “{topic[:60]}”)"]
    if crit:
        return {"level": CRITICAL, "matches": matches + crit}

    caut = [m.group().strip() for _, rx in _CAUT if (m := rx.search(norm))]
    if caut:
        level = CAUTION
    return {"level": level, "matches": matches + caut}


def decide(rule_level, ml_risk=None):
    """Final safety level: the stricter of the rule gate and the ML risk head.
    ML can only escalate, never downgrade a rule hit; a missing model changes nothing."""
    levels = [rule_level] + ([ml_risk] if ml_risk in SEVERITY else [])
    return max(levels, key=SEVERITY.get)


# Fixed crisis wording. Never generated, never tone-shifted.
CRISIS_MESSAGE = (
    "I'm really glad you told me. What you're carrying sounds incredibly heavy, and you deserve "
    "support from a real person right now. I'm an AI and can't keep you safe the way they can.\n\n"
    "If you might act on these thoughts, or you're in danger, please contact emergency services now. "
    "You can also reach a trained listener any time:"
)
CRISIS_RESOURCES = [
    {"name": "Emergency services (India)", "numbers": ["112"],
     "detail": "Nationwide emergency number for police, fire, rescue and medical emergencies."},
    {"name": "Tele-MANAS mental health helpline", "numbers": ["14416", "1-800-891-4416"],
     "detail": "Free, 24/7 counselling in multiple languages.", "url": "https://telemanas.mohfw.gov.in"},
]

# Gemini must never be the source of a contact number or site: it once offered 988 (US) to a user in India.
VERIFIED_NUMBERS = {re.sub(r"\D", "", n) for r in CRISIS_RESOURCES for n in r["numbers"]}
VERIFIED_SITES = {r["url"].split("//")[1] for r in CRISIS_RESOURCES if r.get("url")}
_NUMBER = re.compile(r"\+?\d[\d -]*\d(?!\s*%)")
_SITE = re.compile(r"\b(?:[\w-]+\.)+(?:com|org|net|in|gov|info|co|uk|us)\b", re.I)
_SAFETY_TALK = re.compile(r"\b(helplines?|hotlines?|crisis|lifeline|emergency|are you safe|safe right now|"
                          r"stay safe|keep (?:yourself )?safe)\b", re.I)


def strip_unverified_numbers(reply):
    """Drop every sentence carrying a phone number (3+ digit run) or website that isn't a verified helpline."""
    def unverified(s):
        return (any(re.search(r"\d{3}", n) and re.sub(r"\D", "", n) not in VERIFIED_NUMBERS for n in _NUMBER.findall(s))
                or any(site.lower() not in VERIFIED_SITES for site in _SITE.findall(s)))
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if not unverified(s))


def needs_support(level, reply):
    """Attach the verified helplines to a non-crisis reply on CAUTION, or when the reply itself talks safety."""
    return level == CAUTION or bool(_SAFETY_TALK.search(reply))
