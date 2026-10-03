"""CVPO's observable constraint verdict for ALFWorld.

The WebShop verdict (agent_system/environments/env_package/webshop/envs.py:420) grades a
completed purchase +-1 on a *necessary* condition of success that the agent could have
computed from what it was shown. This is the ALFWorld counterpart, and it keeps the same
two properties:

  * Only agent-visible text enters. The goal sentence is rendered into the prompt's
    {task_description} slot on every step (env_manager.py's ALFWORLD_TEMPLATE), and the
    events below are read off the observation stream the agent already reads. The gamefile
    path is NOT consulted -- ``info['extra.gamefile']`` names the ground-truth object and
    receptacle outright (``pick_and_place_simple-Apple-None-Fridge-...``), which is exactly
    why the earlier ALFWorld CVPO attempt was judged invalid.
  * The verdict is necessary for ``info['won']``: the object type finally placed, the
    receptacle type it was placed in, the number of instances placed, and the heat/cool/clean
    transformation are all conjuncts of the ALFRED goal condition. So a rollout cannot score
    +1 here while failing the task for a reason this check covers, and the term cannot be
    traded against success.

Goal grammar, taken from 128 live train goals (/tmp/alf_goals_probe.py), all six task types:

    put a <obj> in <recep>.                     put some <obj> on <recep>.
    put a clean <obj> in <recep>.               clean some <obj> and put it in <recep>.
    put a hot <obj> in <recep>.                 heat some <obj> and put it in <recep>.
    put a cool <obj> in <recep>.                cool some <obj> and put it in <recep>.
    put two <obj> in <recep>.                   find two <obj> and put them in <recep>.
    look at <obj> under the <lamp>.             examine the <obj> with the <lamp>.

Feedback grammar, observed live rather than read off alfred.twl2 (whose rhs strings carry
debug prefixes and say "in the" where the env actually renders "in/on the"), see
/tmp/alf_feedback_probe.py:

    You put the <obj> <n> in/on the <recep> <m>.
    You pick up the <obj> <n> from the <recep> <m>.
    You heat|cool|clean the <obj> <n> using the <appliance> <m>.
    You turn on the <lamp> <n>.
"""
import re

PUT = re.compile(r'You put the (\w+) (\d+) in/on the (\w+) (\d+)\.')
PICK = re.compile(r'You pick up the (\w+) (\d+) from the (\w+) (\d+)\.')
XFORM = re.compile(r'You (heat|cool|clean) the (\w+) (\d+) using the (\w+) (\d+)\.')
LAMP_ON = re.compile(r'You turn on the (\w+) (\d+)\.')

# "put a hot X in Y" / "put a clean X in Y" / "put a cool X in Y"
_ADJ = {'hot': 'heat', 'cool': 'cool', 'clean': 'clean'}

_PUT_ADJ = re.compile(
    r'^put (?:a|an|some|the) (hot|cool|clean) (\w+) (?:in|on|at) (?:the )?(\w+)\.?$')
_PUT_PLAIN = re.compile(
    r'^put (?:a|an|some|the) (\w+) (?:in|on|at) (?:the )?(\w+)\.?$')
_PUT_TWO = re.compile(
    r'^(?:put|find) two (\w+) (?:in|on|at) (?:the )?(\w+)\.?$')
_VERB_THEN_PUT = re.compile(
    r'^(heat|cool|clean) (?:a|an|some|the) (\w+) and put it (?:in|on|at) (?:the )?(\w+)\.?$')
_FIND_TWO_THEN_PUT = re.compile(
    r'^find two (\w+) and put them (?:in|on|at) (?:the )?(\w+)\.?$')
_LOOK = re.compile(
    r'^(?:look at|examine the) (\w+) (?:under|with) the (\w+)\.?$')


class Constraint:
    """What the goal sentence demands, in terms the observation stream can confirm."""

    __slots__ = ('obj', 'recep', 'transform', 'count', 'lamp')

    def __init__(self, obj, recep=None, transform=None, count=1, lamp=None):
        self.obj = obj
        self.recep = recep
        self.transform = transform
        self.count = count
        self.lamp = lamp

    def __repr__(self):
        return (f'Constraint(obj={self.obj!r}, recep={self.recep!r}, '
                f'transform={self.transform!r}, count={self.count}, lamp={self.lamp!r})')


def parse_goal(goal):
    """Goal sentence -> Constraint, or None if the sentence is not one of the six templates."""
    g = ' '.join(str(goal).strip().lower().split())

    m = _LOOK.match(g)
    if m:
        return Constraint(obj=m.group(1), lamp=m.group(2))

    m = _FIND_TWO_THEN_PUT.match(g) or _PUT_TWO.match(g)
    if m:
        return Constraint(obj=m.group(1), recep=m.group(2), count=2)

    m = _VERB_THEN_PUT.match(g)
    if m:
        return Constraint(obj=m.group(2), recep=m.group(3), transform=m.group(1))

    m = _PUT_ADJ.match(g)
    if m:
        return Constraint(obj=m.group(2), recep=m.group(3), transform=_ADJ[m.group(1)])

    m = _PUT_PLAIN.match(g)
    if m:
        return Constraint(obj=m.group(1), recep=m.group(2))

    return None


class CvpTracker:
    """Accumulates one trajectory's events and reports the +-1 verdict.

    ``verdict()`` returns -1.0 while there is nothing to grade, mirroring the WebShop term's
    "no purchase" case: core_rgpo.py drops rows at -1.0 and centres over the trajectories in
    the batch that do have a verdict, so an ungraded rollout contributes exactly 0.
    """

    __slots__ = ('c', 'placed', 'xformed', 'held', 'any_put', 'lamp_on', 'lamp_ok')

    def __init__(self, goal):
        self.c = parse_goal(goal)
        self.placed = {}       # (obj, idx) -> receptacle type of its most recent put
        self.xformed = set()   # (verb, obj, idx)
        self.held = None       # (obj, idx) currently carried, as far as the text shows
        self.any_put = False
        self.lamp_on = False
        self.lamp_ok = False

    def update(self, obs):
        """Feed one observation string. At most one action feedback per observation, but the
        matches are walked in positional order so a concatenated observation is still correct."""
        text = str(obs)
        events = []
        for m in PICK.finditer(text):
            events.append((m.start(), 'pick', m.groups()))
        for m in PUT.finditer(text):
            events.append((m.start(), 'put', m.groups()))
        for m in XFORM.finditer(text):
            events.append((m.start(), 'xform', m.groups()))
        for m in LAMP_ON.finditer(text):
            events.append((m.start(), 'lamp', m.groups()))
        events.sort(key=lambda e: e[0])

        for _, kind, g in events:
            if kind == 'pick':
                self.held = (g[0], g[1])
            elif kind == 'put':
                obj, idx, recep, _ = g
                self.placed[(obj, idx)] = recep
                self.any_put = True
                if self.held == (obj, idx):
                    self.held = None
            elif kind == 'xform':
                verb, obj, idx = g[0], g[1], g[2]
                self.xformed.add((verb, obj, idx))
            elif kind == 'lamp':
                # look_at_obj_in_light: the light has to be on while the goal object is carried.
                if self.c is not None and self.c.lamp is not None and g[0] == self.c.lamp:
                    self.lamp_on = True
                    if self.held is not None and self.held[0] == self.c.obj:
                        self.lamp_ok = True
        return self.verdict()

    def verdict(self):
        c = self.c
        if c is None:
            return -1.0

        if c.lamp is not None:
            if not self.lamp_on:
                return -1.0
            return 1.0 if self.lamp_ok else 0.0

        if not self.any_put:
            return -1.0
        n = 0
        for (obj, idx), recep in self.placed.items():
            if obj != c.obj or recep != c.recep:
                continue
            if c.transform is not None and (c.transform, obj, idx) not in self.xformed:
                continue
            n += 1
        return 1.0 if n >= c.count else 0.0
