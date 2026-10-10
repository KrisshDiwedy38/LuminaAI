import pytest

from backend.safety import CAUTION, CRITICAL, SAFE, check, decide

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


def test_matches_explain_the_decision():
    assert check("i want to kill myself")["matches"]


def test_decide_takes_the_stricter_level():
    assert decide(SAFE, None) == SAFE  # model missing: rule gate alone decides
    assert decide(SAFE, "garbage") == SAFE
    assert decide(SAFE, CRITICAL) == CRITICAL  # ML can escalate
    assert decide(CRITICAL, SAFE) == CRITICAL  # ML can never downgrade a rule hit
    assert decide(CAUTION, SAFE) == CAUTION
