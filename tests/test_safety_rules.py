import pytest

from backend.safety import CAUTION, CRITICAL, SAFE, check, decide, needs_support, strip_unverified_numbers

CRITICAL_CASES = [
    "I want to kill myself",
    "i wanna die",
    "honestly i just want to die",
    "kms",
    "i'm going to end it all tonight",
    "I don't want to live anymore",
    "i dont wanna be here anymore",
    "everyone would be better off without me",
    "thinking about suicide",
    "I feel suicidal",
    "i want to unalive myself",
    "sewerslide is on my mind",
    "I've been cutting myself again",
    "i want to hurt myself",
    "i took a bunch of pills",
    "i'm going to jump off the bridge",
    "nothing to live for",
    "there's no reason to live",
    "i wrote my goodbye note",
    # misspellings / leetspeak / elongation / separators
    "k1ll mys3lf",
    "i want to kiiiiill myselfff",
    "$uicide",
    "su1c1dal thoughts",
    "i've been thinking about sucide",
    "feeling suicdal",
    "kill-myself",
    "I WANT TO DIE!!!",
    "i want to die lol",  # still critical: hyperbole is not worth a false negative
    # denial does not hide a real signal
    "I'm not suicidal but I want to die",
    "i wanna die now",
    "I don't want to die but I don't want to live either",
    "i dont want to die, i want to kill myself",
    # asking for a method
    "what's the easiest way to die",
    "is there a painless way to end it",
    "how do i kill myself",
    "how to die",
    "how many pills would it take to die",
]

CAUTION_CASES = [
    "I feel so hopeless",
    "i'm worthless",
    "i can't do this anymore",
    "whats the point of anything",
    "nobody cares about me",
    "I'm having a panic attack",
    "i haven't slept in 3 days",
    "I hate myself",
    "i'm such a burden",
    "my dad hits me",
    "i want to kill him lol",
    # explicit denials are downgraded, never to SAFE
    "I'm not suicidal, just really tired",
    "i would never kill myself",
    "i dont wanna die now",
    "I do not want to die",
    "i dont want to kill myself",
    "i never wanna be dead",
    # curiosity about dying: not intent, but worth a gentle check-in
    "how does dying feel ?",
    "what does death feel like",
    "what happens when you die",
    "i keep thinking about death",
]

SAFE_CASES = [
    "I'm having a great day!",
    "my exams are stressing me out",
    "this homework is killing me",
    "i'm dead 💀 that meme",
    "i'm gonna die if i fail this exam",
    "I need to take my pills after dinner",
    "my assignment is overdue",
    "we jumped off the couch",
    "I decided to go for a walk",
    "I'm suited up for the interview",
    "I scored 100% on my quiz in 2 days",
    "i am scared of death",
    "my phone died again",
]


@pytest.mark.parametrize("text", CRITICAL_CASES)
def test_critical(text):
    assert check(text)["level"] == CRITICAL, check(text)


@pytest.mark.parametrize("text", CAUTION_CASES)
def test_caution(text):
    assert check(text)["level"] == CAUTION, check(text)


@pytest.mark.parametrize("text", SAFE_CASES)
def test_safe(text):
    assert check(text)["level"] == SAFE, check(text)


# A vague wish is only as safe as what it refers to: after talk of dying it reads as intent.
@pytest.mark.parametrize("text, recent", [
    ("i wanna feel that", ["i am scared of death", "how does dying feel ?"]),
    ("i want to try it", ["what happens when you die"]),
    ("that sounds peaceful", ["sometimes i think about being dead"]),
    ("i want that too", ["my friend said she wants to die"]),
    ("i wish it was me", ["my cousin died last year"]),
])
def test_follow_up_after_death_talk_is_critical(text, recent):
    r = check(text, recent)
    assert r["level"] == CRITICAL and "right after" in r["matches"][-1], r


@pytest.mark.parametrize("text, recent", [
    ("i wanna feel that", ["the concert last night was amazing"]),
    ("that sounds peaceful", ["i went to the beach today"]),
    ("i wanna feel that", []),
])
def test_follow_up_without_death_talk_is_safe(text, recent):
    assert check(text, recent)["level"] == SAFE


def test_unverified_helplines_are_removed_from_replies():
    reply = ("Whoa, hold on. I care about you. Please call or text 988 if you are in the US. "
             "You can also visit findahelpline.com. You don't have to carry this alone.")
    out = strip_unverified_numbers(reply)
    assert "988" not in out and "findahelpline" not in out
    assert out == "Whoa, hold on. I care about you. You don't have to carry this alone."


def test_verified_helplines_and_ordinary_numbers_stay():
    reply = ("You can call Tele-MANAS on 14416 or 1-800-891-4416, or 112 in an emergency. "
             "Try 4-7-8 breathing for 10 minutes. Your feelings are 100% valid.")
    assert strip_unverified_numbers(reply) == reply


def test_support_shown_on_caution_or_when_reply_talks_safety():
    assert needs_support(CAUTION, "That sounds heavy.")
    assert needs_support(SAFE, "I'm worried about you. Are you safe right now?")
    assert not needs_support(SAFE, "Ah, physics. Which chapter is it?")


def test_matches_explain_the_decision():
    assert check("i want to kill myself")["matches"]


def test_decide_takes_the_stricter_level():
    assert decide(SAFE, None) == SAFE  # model missing: rule gate alone decides
    assert decide(SAFE, "garbage") == SAFE
    assert decide(SAFE, CRITICAL) == CRITICAL  # ML can escalate
    assert decide(CRITICAL, SAFE) == CRITICAL  # ML can never downgrade a rule hit
    assert decide(CAUTION, SAFE) == CAUTION
