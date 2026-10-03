# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import ray
import gym
import numpy as np

try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover - rapidfuzz ships with the webshop env
    _rf_fuzz = None


def _fuzz_ratio(a: str, b: str) -> float:
    """Token-set similarity, matching how ``goal.py`` compares option values to buttons.

    WebShop's own ``get_reward`` accepts an option when ``fuzz.token_set_ratio > 85``, so the
    observable check must use the same comparison or it would disagree with the reward it predicts.
    """
    if _rf_fuzz is None:
        return 100.0 if a == b else 0.0
    return float(_rf_fuzz.token_set_ratio(a, b))


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs, min_products_before_buy: int = 0,
                 caa_std_floor: float = 0.1, caa_scope: str = 'click',
                 evidence_chars: int = 0):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        from web_agent_site.engine.goal import get_reward  # noqa: WPS433 (runtime import)

        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        self._get_reward = get_reward
        self._sid = None
        self._goal_idx = None
        self._goal_key = None
        # Compare-before-commit gate. 0 disables it and makes this class byte-identical in behaviour
        # to the pre-gate version.
        self._min_products_before_buy = int(min_products_before_buy or 0)
        self._caa_std_floor = float(caa_std_floor)
        # 'click' scores the product the agent clicked against the results page it was shown; 'buy'
        # scores the completed purchase against every candidate the episode ever saw. The measured
        # ceiling of a *perfect* results-page rule is SR 0.7100 (attributes are 88.3% invisible until
        # a product is opened), so 'click' rewards a decision that cannot reach the 0.7812 target and
        # implicitly rewards buying the first result. 'buy' rewards the outcome of comparing instead.
        self._caa_scope = str(caa_scope or 'click')
        # >0 keeps the text of the sub-pages the agent actually read, so a later comparison has the
        # evidence in front of it. 0 keeps the prompt byte-identical to the pre-feature version.
        self._evidence_chars = int(evidence_chars or 0)
        self._viewed = []       # distinct asins opened this episode, in the order they were opened
        self._seen = set()      # every asin shown to the agent on a search-results page
        self._evidence = {}     # asin -> {sub-page name: text the agent read there}
        self._blocked_buys = 0
        self._score_cache = {}  # asin -> best achievable dense score for the *current* goal
        self._caa_regret_sum = 0.0
        self._caa_clicks = 0
        self._bought_ok = 0.0   # 1.0 iff the purchased product offered every option value asked for

    def _req_scores(self):
        """The per-requirement sub-scores of the *finished* episode, or None.

        ``get_reward(..., verbose=True)`` decomposes the WebShop score into
        ``r_type * (w_att*r_att + w_option*r_option + w_price*r_price)`` and SimServer.done stores
        that dict in ``user_sessions[sid]['verbose_info']``. Only reachable once the agent has
        clicked "Buy Now"; a timed-out episode has no entry.

        ``self._sid`` (captured in reset) is required rather than ``self.env.session``:
        WebAgentTextEnv.step calls ``self.reset()`` when the episode terminates, which replaces
        ``session`` with a fresh random id whose session dict has no ``verbose_info`` -- reading it
        would silently return None on every successful purchase.
        """
        if self._sid is None:
            return None
        session = self.env.server.user_sessions.get(self._sid)
        if not session:
            return None
        return session.get('verbose_info')

    def _session(self):
        if self._sid is None:
            return {}
        return self.env.server.user_sessions.get(self._sid) or {}

    def _sync_viewed(self):
        """Append the currently opened product to the ordered list of products read this episode.

        ``session['asins']`` already accumulates every opened asin and -- unlike ``session['asin']``
        and ``session['options']``, which ``search_results`` clears on every new search
        (web_agent_text_env.py:373-374) -- it survives re-searching. It is a *set* though, so the
        order in which the agent read the products is kept here instead.
        """
        asin = self._session().get('asin')
        if asin and asin not in self._viewed:
            self._viewed.append(asin)

    def _sync_seen(self):
        """Accumulate every asin the agent has been *shown* in a search-results page.

        The results list is not stored in the session (``search_results`` renders it straight to
        HTML), but the clickables of a results page are exactly the asins on it, lowercased. This
        separates the two ways an episode can fail -- the target was never retrieved, or it was
        retrieved and the agent bought something else -- which no existing metric distinguishes.
        """
        for c in (self.env.get_available_actions().get('clickables') or []):
            if len(c) == 10 and c[0] in 'bB' and c.isalnum():
                self._seen.add(c.upper())

    def _target_asin(self):
        goal = self._session().get('goal') or {}
        return goal.get('asin')

    # ------------------------------------------------------------------
    # Counterfactual action-set values ---------------------------------
    # ------------------------------------------------------------------
    #
    # ``goal.py:get_reward(product, goal, price, options)`` is a pure function, so the score of every
    # product on a results page can be evaluated for actions the agent did *not* take. Measured on
    # 200 val goals with the instruction as query: a policy that opens rank 1 and buys it tops out at
    # SR 0.510 / dense 0.764, while choosing the best of the ten reaches 0.885 / 0.960. That 0.20
    # dense gap is what a single binary terminal reward has to teach through, from 8 rollouts of one
    # goal -- and the measured collapse to 5.85-step episodes says it does not. These values turn the
    # same information into a per-click signal that exists on every results page of every trajectory,
    # whether or not the episode ended in a purchase.

    def _page_asins(self):
        """Uppercased asins on the page the agent is looking at right now.

        Empty off a results page: the clickables of a product page are its options plus navigation.
        """
        return [c.upper() for c in (self.env.get_available_actions().get('clickables') or [])
                if len(c) == 10 and c[0] in 'bB' and c.isalnum()]

    def _best_score(self, asin, goal):
        """Best dense score buying ``asin`` could yield: cheapest variant, best option assignment.

        Passing every option *value* at once is the exact maximum rather than a loose bound: a
        purchase picks one value per category, and the synthetic goals are an ``itertools.product``
        over categories, so a goal names at most one option per category and one value can satisfy
        at most that one.
        """
        if asin in self._score_cache:
            return self._score_cache[asin]
        server = self.env.server
        product = (getattr(server, 'product_item_dict', {}) or {}).get(asin)
        score = 0.0
        if product is not None:
            price = (getattr(server, 'product_prices', {}) or {}).get(asin, 1e6)
            if isinstance(price, dict):
                price = min(price.values()) if price else 1e6
            options = {}
            for cat, values in (product.get('customization_options') or {}).items():
                for j, value in enumerate(values or []):
                    options[f'{cat}_{j}'] = value['value'] if isinstance(value, dict) else value
            try:
                score = float(self._get_reward(product, goal, price, options))
            except Exception:
                score = 0.0
        self._score_cache[asin] = score
        return score

    def _click_target(self, action):
        m = re.match(r'\s*click\s*\[\s*(.+?)\s*\]\s*$', str(action), flags=re.IGNORECASE)
        return m.group(1).strip().upper() if m else None

    def _choice_advantage(self, action, candidates):
        """How good the clicked product was relative to the page the agent was shown.

        Standardised *within the page* -- the comparison the agent actually faced -- with a floor on
        the spread so a page of near-identical candidates cannot manufacture a large advantage out of
        rounding. Returns ``(advantage, regret, applicable)``; ``regret`` is how much dense score the
        best candidate on that page would have added, and is the metric that says whether the term is
        working at all.
        """
        if len(candidates) < 2:
            return 0.0, 0.0, False
        chosen = self._click_target(action)
        if chosen is None or chosen not in candidates:
            return 0.0, 0.0, False
        goal = self._session().get('goal') or {}
        if not goal:
            return 0.0, 0.0, False
        ys = np.array([self._best_score(a, goal) for a in candidates], dtype=np.float64)
        y = float(ys[candidates.index(chosen)])
        adv = (y - float(ys.mean())) / max(float(ys.std()), self._caa_std_floor)
        return float(adv), float(ys.max() - y), True

    def _purchase_advantage(self, realized_score):
        """How good the completed purchase was against every candidate the episode ever saw.

        The click-scope version grades a decision taken *before* the product is opened, and a perfect
        rule on that information caps out at SR 0.7100 -- below the target -- because 88.3% of a
        goal's attributes only exist in Attributes/Features/Description. Grading the purchase instead
        rewards whatever the agent did to inform it, and because ``realized_score`` uses the options
        the agent actually selected while the reference scores use each candidate's best possible
        options, it also grades the option and price choice rather than the product choice alone.

        Nonzero for every completed purchase, independently of whether any other rollout in the group
        succeeded, so it cannot be silenced by a uniformly-failing group.
        """
        goal = self._session().get('goal') or {}
        if not goal or len(self._seen) < 2:
            return 0.0, 0.0, False
        ys = np.array([self._best_score(a, goal) for a in sorted(self._seen)], dtype=np.float64)
        y = float(realized_score)
        adv = (y - float(ys.mean())) / max(float(ys.std()), self._caa_std_floor)
        return float(adv), float(max(ys.max() - y, 0.0)), True

    #: Product sub-pages worth remembering. A product page offers exactly
    #: ``description``/``features``/``reviews``; reading description + features lifts the fraction of
    #: goal attributes that are literally visible from 0.1204 to 0.5554, while ``reviews`` averages
    #: 221 chars of opinion and carries no attribute evidence, so it is left out of the budget.
    _SUBPAGES = ('description', 'features')

    def _record_evidence(self, action, obs):
        """Keep the text of a sub-page the agent just read, keyed by the product it belongs to.

        One inspect cycle costs 4 steps and ``history_length`` is 2, so by the time a second candidate
        is opened the first one's description has left the context and no comparison is representable.
        This is a replay of pages the agent was already shown -- it adds no unobserved information.
        """
        if self._evidence_chars <= 0:
            return
        page = self._click_target(action)
        if page is None or page.lower() not in self._SUBPAGES:
            return
        asin = self._session().get('asin')
        if not asin:
            return
        text = str(obs)
        instruction = ((self._session().get('goal') or {}).get('instruction_text') or '')
        if instruction and instruction in text:
            # Every observation is prefixed with the instruction; keeping it would spend the budget
            # re-quoting text the prompt already contains.
            text = text[text.index(instruction) + len(instruction):]
        text = ' '.join(text.replace('[SEP]', ' ').split())
        # Sub-pages open with their navigation bar, which is the same on every one of them and would
        # spend the front of the budget -- the part most likely to survive truncation -- on nothing.
        for nav in ('Back to Search', '< Prev'):
            if text.startswith(nav):
                text = text[len(nav):].lstrip()
        text = text[:self._evidence_chars]
        if text:
            self._evidence.setdefault(asin, {})[page.lower()] = text

    @staticmethod
    def _is_buy_action(action) -> bool:
        return str(action).strip().lower().replace(' ', '') == 'click[buynow]'

    def _required_option_values(self):
        """The option values the instruction names, e.g. ['coal grey', '36w x 32l'].

        Nothing privileged: WebShop prints these verbatim in the instruction ("with color: coal grey,
        and size: 36w x 32l"), which is in the prompt on every single step. ``goal_options`` is simply
        the parsed form of that same sentence.
        """
        goal = self._session().get('goal') or {}
        opts = goal.get('goal_options') or {}
        values = opts.values() if isinstance(opts, dict) else opts
        return [str(v).strip().lower() for v in values if str(v).strip()]

    def _offers_required_options(self, asin):
        """Does this product's option buttons offer every value the instruction asks for?

        Returns ``None`` when the instruction names no option, so the caller can tell "nothing to
        verify" apart from "failed". This is decidable from a single observation: ``click[asin]`` opens
        with the full button list (17 values on average) ahead of the title, so no sub-page, no
        truncation and no cross-page memory is involved -- unlike the description/features route, which
        the base model cannot use (handed all ten candidates with their prose it picks at SR 0.3400,
        below always-rank-1's 0.5100).

        Measured over 200 val goals: 2.37 of 10 page-1 candidates pass, the target survives 0.9945 of
        the time, rank 1 passes only 0.6500, and buying the first candidate that passes scores
        SR 0.7300 against 0.5100 for buying rank 1.
        """
        wanted = self._required_option_values()
        if not wanted:
            return None
        product = (getattr(self.env.server, 'product_item_dict', {}) or {}).get(asin) or {}
        have = []
        for values in (product.get('customization_options') or {}).values():
            for v in values or []:
                have.append(str(v['value'] if isinstance(v, dict) else v).strip().lower())
        for want in wanted:
            if not any(want in h or h in want or _fuzz_ratio(want, h) > 85 for h in have):
                return False
        return True

    def _gate_message(self) -> str:
        return (
            f"Your click on Buy Now was rejected: you must open and read at least "
            f"{self._min_products_before_buy} different products before buying, and you have only "
            f"opened {len(self._viewed)} so far. Click a product link to read another product, then buy "
            f"the one that matches the instruction best."
        )

    def _state_info(self) -> dict:
        """Everything the agent itself has already seen, in a form it can be reminded of.

        Titles and prices come from ``product_item_dict``, but only for asins the agent *opened*, so
        this reports back what it already read rather than adding unobserved information.
        """
        session = self._session()
        target = self._target_asin()
        viewed = []
        item_dict = getattr(self.env.server, 'product_item_dict', {}) or {}
        for asin in self._viewed:
            info = item_dict.get(asin) or {}
            viewed.append({
                'asin': asin,
                'title': info.get('Title') or info.get('title') or '',
                'price': info.get('Price') or info.get('price') or '',
                # Only the sub-pages this episode actually opened, so the recap can never mention a
                # page the agent has not read.
                'evidence': dict(self._evidence.get(asin) or {}),
            })
        return {
            'selected_options': dict(session.get('options') or {}),
            'viewed_products': viewed,
            'n_viewed': len(self._viewed),
            'blocked_buys': self._blocked_buys,
            'n_seen': len(self._seen),
            'n_pages_read': sum(len(v) for v in self._evidence.values()),
            'target_seen': bool(target and target in self._seen),
            'target_opened': bool(target and target in self._viewed),
            # Whether the product this episode bought offers every option value the instruction names.
            # 0.0 until a purchase happens, so it is safe to read off any row.
            'bought_options_ok': float(self._bought_ok),
        }

    def step(self, action):
        """Execute a step in the environment"""
        # Captured *before* the action, because the action set the agent chose from is the one on the
        # page it was looking at, not the page the click navigates to.
        candidates = self._page_asins()
        gated = (
            self._min_products_before_buy > 0
            and self._is_buy_action(action)
            and len(self._viewed) < self._min_products_before_buy
        )
        if gated:
            # Do not hand the action to the env: the purchase must not happen. The page is unchanged,
            # so the observation is the current one plus a constant explanation of the rejection. The
            # message is identical for every row in the same situation, so anchor_obs (GiGPO's
            # step-grouping key) stays shared across trajectories instead of becoming unique.
            self._blocked_buys += 1
            obs = self.env.observation + ' [SEP] ' + self._gate_message()
            reward, done, info = 0, False, None
        else:
            obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        self._sync_viewed()
        self._sync_seen()
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward
        info['req_scores'] = self._req_scores() if done else None
        info['goal_idx'] = self._goal_idx
        info['goal_key'] = self._goal_key
        info['buy_blocked'] = bool(gated)
        self._record_evidence(action, obs)
        if self._caa_scope == 'buy':
            # Graded once, on the step that completes the purchase: ``reward`` at ``done`` is the
            # env's dense score for what was actually bought.
            if done and not gated:
                caa_adv, caa_regret, caa_ok = self._purchase_advantage(reward)
            else:
                caa_adv, caa_regret, caa_ok = 0.0, 0.0, False
        else:
            caa_adv, caa_regret, caa_ok = self._choice_advantage(action, candidates)
        if caa_ok:
            self._caa_regret_sum += caa_regret
            self._caa_clicks += 1
        info['caa_adv'] = caa_adv
        info['caa_regret'] = caa_regret
        info['caa_applicable'] = caa_ok
        info['caa_n_candidates'] = len(candidates)
        # Episode-level aggregate, because _process_batch only reads the terminal row: the mean dense
        # score the agent gave up on the results pages it clicked from. This is the number the term is
        # supposed to drive down, and it is measurable from step 1 with no successful purchase.
        info['caa_mean_regret'] = (self._caa_regret_sum / self._caa_clicks) if self._caa_clicks else 0.0
        info['caa_clicks'] = self._caa_clicks
        # The constraint verdict on the completed purchase: 1.0 offers every option value the
        # instruction names, 0.0 does not, -1.0 nothing to grade (no purchase on this row, or a goal
        # that names no option -- 1.5% of goals). Unlike caa this consults no reward function and does
        # not need the target's identity; it compares two things the agent was shown, the instruction
        # and the option buttons, so it is computable by the agent itself.
        cvp = -1.0
        if done and not gated:
            verdict = self._offers_required_options(str(self._session().get('asin') or '').upper())
            if verdict is not None:
                cvp = float(verdict)
                self._bought_ok = cvp
        info['cvp_pass'] = cvp
        info.update(self._state_info())

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        # SimServer.user_sessions keeps every session it has ever served (goal, page state,
        # verbose_info, asins, ...) and is never pruned, so it grows without bound over a long
        # training run. The new session is recreated by receive() during reset(), so dropping the
        # history here is safe.
        self.env.server.user_sessions.clear()
        obs, info = self.env.reset(session=idx)
        self._viewed = []
        self._seen = set()
        self._evidence = {}
        self._blocked_buys = 0
        self._score_cache = {}   # keyed by asin only, so it must not outlive the goal
        self._caa_regret_sum = 0.0
        self._caa_clicks = 0
        self._bought_ok = 0.0
        # Read the id back from the env: reset() stringifies idx and may prepend session_prefix,
        # so str(idx) is not always the user_sessions key.
        self._sid = self.env.session
        # The goal index *is* the goal identity: reset(session=idx) selects the instruction. It has
        # to travel in info because the observation the agent sees has the instruction stripped out
        # (WebshopEnvironmentManager.format_obs), so no downstream text distinguishes two goals.
        self._goal_idx = int(idx)
        self._goal_key = self._build_goal_key()
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        info['req_scores'] = None
        info['goal_idx'] = self._goal_idx
        info['goal_key'] = self._goal_key
        info['buy_blocked'] = False
        info['caa_adv'] = 0.0
        info['caa_regret'] = 0.0
        info['caa_applicable'] = False
        info['caa_n_candidates'] = 0
        info['caa_mean_regret'] = 0.0
        info['caa_clicks'] = 0
        info['cvp_pass'] = -1.0
        info.update(self._state_info())
        return obs, info

    def _build_goal_key(self):
        """A '|'-separated hierarchical difficulty key for this goal, most specific level first.

        Level 0 is the target product (``asin``): the synthetic goal set is the cartesian product of
        each product's options, so ~6410 train goals cover only ~372 products and one product is
        revisited every ~23 steps -- whereas an individual goal is drawn at most once in a 150-step
        run, which is why keying on the goal itself leaves the difficulty term permanently in warmup.
        Level 1 is the requirement profile ``natt-nopt-price``, which is dense from the second step
        and is what actually drives difficulty (every attribute and option is one more thing that
        must match). Level 1 also carries over to goals whose product has never been seen.
        """
        session = self.env.server.user_sessions.get(self._sid) or {}
        goal = session.get('goal') or {}
        if not goal:
            return None
        n_att = len(goal.get('attributes') or [])
        n_opt = len(goal.get('goal_options') or [])
        has_price = int(float(goal.get('price_upper', 1e6)) < 1e6)
        return f"{goal.get('asin')}|{n_att}-{n_opt}-{has_price}"
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        min_products_before_buy: int = 0,
        caa_std_floor: float = 0.1,
        caa_scope: str = 'click',
        evidence_chars: int = 0,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs,
                                       min_products_before_buy, caa_std_floor,
                                       caa_scope, evidence_chars)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
        return self.reset_to(idx)

    def reset_to(self, goal_idxs):
        """Reset every worker to a *caller-specified* goal index instead of drawing fresh ones.

        Needed by CRB (``recipe/sgpo/rewind.py``), which has to put the environments back on the
        goals a finished rollout used before it can replay that rollout's action prefixes. Worker
        ``i`` keeps the seed it was constructed with (``seed + i // group_n``), so a prefix recorded
        on one row replays exactly only on a row of the *same* group -- the condition
        ``recipe/cbpo/determinism_check.py`` verifies against the real environment.
        """
        if len(goal_idxs) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} goal indices, got {len(goal_idxs)}',
            )

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, goal_idxs):
            future = worker.reset.remote(int(i))
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    min_products_before_buy: int = 0,
    caa_std_floor: float = 0.1,
    caa_scope: str = 'click',
    evidence_chars: int = 0,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        min_products_before_buy=min_products_before_buy,
        caa_scope=caa_scope,
        evidence_chars=evidence_chars,
        caa_std_floor=caa_std_floor,
    )