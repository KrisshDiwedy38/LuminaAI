from train import is_better


def run(epochs, tol):
    """epochs: [(critical_recall, score)]. Returns the 1-based epoch that would be saved."""
    best, max_cr, saved = None, 0.0, None
    for n, (cr, score) in enumerate(epochs, 1):
        max_cr = max(max_cr, cr)
        if is_better(best, cr, score, max_cr, tol):
            best, saved = (cr, score), n
    return saved


def test_one_crisis_example_is_a_tie():
    # the real 4-epoch run (65 val crises): epoch 4 is better everywhere but one example lower on recall
    epochs = [(34 / 65, 0.676), (56 / 65, 1.166), (54 / 65, 1.279), (55 / 65, 1.324)]
    assert run(epochs, 1 / 65 + 1e-9) == 4


def test_two_crisis_examples_still_lose():
    assert run([(56 / 65, 1.0), (54 / 65, 9.0)], 1 / 65 + 1e-9) == 1


def test_small_drops_cannot_add_up():
    # each epoch is one example below the previous one: only the first stays within one example of the best
    assert run([(0.90, 1.0), (0.885, 2.0), (0.87, 3.0), (0.855, 4.0)], 0.0154) == 2


def test_clearly_higher_recall_wins():
    assert run([(0.80, 2.0), (0.90, 1.0)], 1 / 65) == 2
