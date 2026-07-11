from __future__ import annotations

import unittest
from types import SimpleNamespace

from a_top10.steps.step4_theme_boost import _get_dragon_bonus, _get_rank_decay_k


class ThemeConfigTests(unittest.TestCase):
    def test_theme_block_controls_rank_decay(self) -> None:
        settings = SimpleNamespace(theme=SimpleNamespace(sigmoid_k=0.27))
        self.assertEqual(_get_rank_decay_k(settings), 0.27)

    def test_dragon_bonus_can_be_disabled(self) -> None:
        settings = SimpleNamespace(
            theme=SimpleNamespace(use_dragon_leader_rank=False, dragon_bonus=0.5)
        )
        self.assertEqual(_get_dragon_bonus(settings), 0.0)


if __name__ == "__main__":
    unittest.main()
