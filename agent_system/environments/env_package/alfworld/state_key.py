"""Sound state key sigma(s) for ALFWorld -- the input to SGPO's state-graph advantage.

sigma = hash(canon(admissible_commands), progress_multiset)

Both halves are things the agent already sees verbatim:
  - canon(admissible_commands): the env's own admissible-action list, sorted and with 'help'
    dropped. This is a function of the world's real state (location, inventory -- 'put X in Y'
    is only admissible while holding X -- open/closed receptacles), and it is already rendered
    into the prompt's `{admissible_actions}` slot (env_manager.py's build_text_obs). Zero extra
    disclosure.
  - progress_multiset: state-change events (heat/cool/clean/turn-on/slice) read off the
    observation stream. admissible_commands does not change when an object already in view gets
    heated/cooled/cleaned/sliced/lit, so this multiset is what recovers that missing state bit.
    The grammar below is the one validated live against the game's own feedback strings in
    cvp.py (verified in /tmp/alf_probe.py); unlike cvp.py this module does NOT read the gamefile,
    does NOT build a checklist, and does NOT compare against a task line -- it only tracks what
    happened, not whether it was the right thing to happen.

Measured properties (see /tmp/sgpo_measure.py, /tmp/sgpo_alias_diag.py, /root/.claude/plans/
state-graph-sgpo.md section 6): world-state-only aliasing ~11.67% (not the <2% originally hoped
for), same-(sigma,a)-diverges-to-different-sigma' ~0.000-0.007 (reliable transition function),
sigma merges GiGPO's over-split text_obs key rather than shrinking groups the way HGPO's history
prefixes do. The residual aliasing is priced into the DP as a pessimism term (core_sgpo.py),
not filtered out here.
"""
import re
from collections import Counter

DROP = frozenset({'help', 'look', 'inventory'})

STATE = re.compile(r'You (heat|cool|clean) the (\w+) (\d+) using the (\w+) (\d+)\.')
LAMP = re.compile(r'You turn on the (\w+) (\d+)\.')
SLICE = re.compile(r'You sliced the (\w+) (\d+) with the (\w+) (\d+)\.')


def canon(admissible_commands):
    """Sorted tuple of admissible actions with help/look/inventory dropped."""
    return tuple(sorted(c for c in admissible_commands if c not in DROP))


def progress_events(obs):
    """Multiset (as a sorted tuple) of state-change events visible in one observation string."""
    ev = []
    for verb, obj, n, _recep, _rn in STATE.findall(obs):
        ev.append((verb, obj, n))
    for obj, n in LAMP.findall(obs):
        ev.append(('lamp', obj, n))
    for obj, n, _knife, _kn in SLICE.findall(obs):
        ev.append(('slice', obj, n))
    return ev


class ProgressTracker:
    """Accumulates the progress multiset across one trajectory's observation stream."""

    def __init__(self):
        self._counter = Counter()

    def update(self, obs):
        self._counter.update(progress_events(obs))
        return self.multiset()

    def multiset(self):
        return tuple(sorted(self._counter.elements()))


def sound_key(admissible_commands, progress_multiset):
    """sigma(s) as a hashable tuple; callers that need a fixed-type array should str() this."""
    return (canon(admissible_commands), progress_multiset)
