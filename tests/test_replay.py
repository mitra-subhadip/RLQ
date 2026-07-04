import numpy as np

from rl_qtranspiler.environment import PlacementState
from rl_qtranspiler.replay import PrioritizedReplayBuffer, Transition


def test_prioritized_replay_sample_and_update():
    buffer = PrioritizedReplayBuffer(capacity=16)
    state = PlacementState((-1, -1), (-1, -1, -1), 0)
    next_state = PlacementState((0, -1), (0, -1, -1), 1)
    for index in range(10):
        buffer.add(
            Transition("p", state, index % 3, -0.1, next_state, False),
            priority=index + 1,
        )
    batch = buffer.sample(4, beta=0.4)
    assert len(batch.transitions) == 4
    assert np.all(batch.importance_weights > 0)
    assert np.max(batch.importance_weights) <= 1
    buffer.update_priorities(batch.tree_indices, np.ones(4))
