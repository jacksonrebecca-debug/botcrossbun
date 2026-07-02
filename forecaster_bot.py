"""
=============================================================================
 GLASS-BOX FORECASTER — the live port  (forecaster_bot.py)
=============================================================================
A tournament bot for the Metaculus FutureEval / MiniBench competition, built by
porting the layers you developed and validated in the browser glass box.

WHAT THE FRAMEWORK ALREADY GIVES YOU (so we do NOT rebuild it):
  - Metaculus auth, question loading, posting forecasts + comments, the cron loop
  - The ENSEMBLE: it runs your forecast function `predictions_per_research_report`
    times automatically  (your v4 "five passes")
  - AGGREGATION: it combines those passes  (binary default is already MEDIAN;
    MC default is mean — we override that below)
  - structure_output(): an LLM-based parser that turns messy prose into a typed
    object  (exactly your v3 insight — you reinvented this, which is why it ports
    to "nothing", it's native)

WHAT WE PORT FROM THE GLASS BOX (the genuine value-add):
  1. OPERATIONALISE-AND-BIND  (your v6/v7 crown jewel) — folded into run_research
     so it is computed ONCE and shared across every ensemble pass. This is the
     fix that made Mythos flip: the bot reads each official option's definition
     the way the resolver wrote it, instead of improvising.
  2. A TIME-AWARE PRIOR  (your v5 experiment) — status quo by default, but weight
     the trajectory when a deadline sits inside the resolution window.
  3. MEDIAN AGGREGATION FOR MC  (the No-Stream post-mortem lesson) — mean dilutes
     a correct minority on high-disagreement questions; median is more robust.
  4. A CONSISTENCY CHECK  (your "Forecaster 1" catch, automated) — logged as a
     diagnostic so a words-vs-numbers slip is visible in the bot's comment.

LEAN CONFIG: 3 passes, not 5 — halves cost with little accuracy loss, because the
biggest score lever is the MODEL, not the pass count (per the FutureEval docs).

Run:  python forecaster_bot.py --mode test_questions   (the unscored arena first)
      python forecaster_bot.py --mode tournament       (live: MiniBench + seasonal)
=============================================================================
"""

import argparse
import asyncio
import itertools
import logging
import re
import statistics
from datetime import datetime

import dotenv
import httpx

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    BinaryPrediction,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# The one prompt fragment that encodes your v5 finding. Note it is conditional:
# the status-quo heuristic is right for slow questions and wrong for one with a
# hard deadline inside the window (the Fable/Mythos case). The prior must flex
# with the question's TIME STRUCTURE — your very first insight.
TIME_AWARE_PRIOR = clean_indents(
    """
    Weight the status quo by default, since the world changes slowly most of the time.
    BUT if a scheduled event or hard deadline inside the resolution window is likely
    to change the outcome (a ruling, an expiry, a planned launch, a stated deadline),
    weight the trajectory toward that change instead. Ask explicitly whether the window
    is long enough, and the scheduled events strong enough, for the status quo to move
    before the question resolves.
    """
)


# --- COVERAGE-RECENCY SIGNAL --------------------------------------------------
# A weak PROXY for "is this process clustering right now?". AskNews stamps each
# article with "Publish date: <Month DD, YYYY HH:MM AM>"; we pull those out and
# bucket by age. IMPORTANT HONESTY: AskNews always returns ~6 fresh "hot"
# articles by design, and its historical search is relevance-ranked (which skews
# recent), so raw recency ALWAYS looks accelerating. We therefore present an age
# histogram + a soft skew read, explicitly flagged as a biased proxy — never a
# verdict. It is an input to the model's judgement, exactly like the Poisson rail.
_PUB_DATE = re.compile(r"Publish date:\s*([A-Z][a-z]+ \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M)")


def _parse_pub_dates(text: str) -> list[datetime]:
    dates = []
    for m in _PUB_DATE.findall(text or ""):
        try:
            dates.append(datetime.strptime(m, "%B %d, %Y %I:%M %p"))
        except ValueError:
            pass
    return dates


def _coverage_recency_note(research_text: str, now: datetime) -> str:
    dates = _parse_pub_dates(research_text)
    if len(dates) < 4:
        return ""  # too few dated articles to say anything — stay silent, don't mislead
    b = {"0-7d": 0, "8-30d": 0, "31-90d": 0, ">90d": 0}
    for d in dates:
        age = (now - d).days
        if age <= 7:
            b["0-7d"] += 1
        elif age <= 30:
            b["8-30d"] += 1
        elif age <= 90:
            b["31-90d"] += 1
        else:
            b[">90d"] += 1
    recent, older = b["0-7d"] + b["8-30d"], b["31-90d"] + b[">90d"]
    if older == 0:
        skew = "entirely recent (but that may just be AskNews's fresh-article bias — weak signal)"
    elif recent > 2 * older:
        skew = "pronounced toward recent (weak evidence of a possible burst, IF self-exciting)"
    elif recent > older:
        skew = "recent-leaning"
    else:
        skew = "spread out — no recent clustering"
    hist = ", ".join(f"{k}: {v}" for k, v in b.items())
    return clean_indents(
        f"""

        --- COVERAGE RECENCY SIGNAL (a weak, retrieval-biased PROXY — not the event rate) ---
        Ages of retrieved articles: {hist}.
        Read: {skew}.
        Caveats: AskNews over-samples fresh articles by design, and media over-covers dramatic
        one-offs. So treat only a PRONOUNCED recent skew as weak evidence of a burst, only for a
        self-exciting process, and never adjust the rate on this signal alone.
        """
    )


# =============================================================================
#  UPGRADE 1 — REASON FROM CLEAN DATA  (the resolution-source reader)
#  Inspired by last season's winner (GreeneiBot2), which reads the actual page a
#  question resolves against. Your own "coverage" lesson: the resolution SOURCE
#  is not the modelling source. Many questions resolve off ONE specific page
#  (a gov table, a tracker, a Wikipedia figure) whose URL sits in the criteria
#  text. This block extracts those URLs, fetches them, strips the HTML to text,
#  and drops the real page into the shared research so all passes read the
#  ground truth, not just news about it. Toggle: remove the one call in
#  run_research to disable. Free — no Exa/paid fetcher, just httpx.
# =============================================================================
_URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_resolution_urls(question) -> list[str]:
    blob = " ".join(
        s for s in [question.resolution_criteria, question.fine_print, question.background_info] if s
    )
    urls, seen = [], set()
    for u in _URL_RE.findall(blob):
        u = u.rstrip(".,);'\"")
        low = u.lower()
        if "metaculus.com" in low:  # skip self-links back to the question page
            continue
        if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip")):
            continue
        if u not in seen:
            seen.add(u)
            urls.append(u)
    return urls[:2]  # cap at 2 — don't fan out into the whole internet


async def _fetch_url_text(url: str, limit: int = 3000) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; forecasting-bot)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
        text = _SCRIPT_RE.sub(" ", html)   # drop script/style blocks
        text = _TAG_RE.sub(" ", text)      # strip remaining tags
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]
    except Exception:
        return ""  # a bad/paywalled/JS page must never break research


async def _resolution_source_note(question) -> str:
    urls = _extract_resolution_urls(question)
    if not urls:
        return ""
    chunks = []
    for u in urls:
        body = await _fetch_url_text(u)
        if body:
            chunks.append(f"[{u}]\n{body}")
    if not chunks:
        return ""
    joined = "\n\n".join(chunks)
    return clean_indents(
        f"""

        --- RESOLUTION SOURCE CONTENT (fetched from links in the criteria — the page this resolves against) ---
        {joined}
        --- end resolution source (note: static fetch only; JS-heavy or paywalled pages may be thin) ---
        """
    )


# =============================================================================
#  UPGRADE 2 — DIVERSE-MODEL ENSEMBLE  ("three passes from separate SOTA")
#  The 3 ensemble passes normally all run the SAME model = three takes from one
#  mind. Real diversity (the "wisdom of deliberating AI crowds" finding) needs
#  DIFFERENT model families, so the disagreement carries signal, not just
#  temperature noise. This pool rotates one pass per family. Toggle off ->
#  falls back to the single "default" model.
#  NOTE: verify these exact model IDs against openrouter.ai/models — provider
#  names drift; these are the right shape, not guaranteed current strings.
# =============================================================================
DIVERSE_ENSEMBLE = True
FORECASTER_POOL = [
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/openai/gpt-5",
    "openrouter/google/gemini-2.5-pro",
]

# =============================================================================
#  UPGRADE 3 — ReAct RESEARCH LOOP  ("keep a reason loop open for more data")
#  One-shot research takes a single snapshot. A ReAct loop instead: search ->
#  ask "what's still MISSING?" -> search for that gap -> repeat. smingers reported
#  this was his single biggest score gain. THE RISK: a loop that keeps deciding it
#  needs more will keep spending your AskNews quota. So MAX_REACT_STEPS is a HARD
#  cap it can never exceed, and the loop also stops early the moment the model says
#  the research is sufficient. Toggle off -> pure one-shot research.
#  QUOTA MATH: each step is one more AskNews call, so a question costs up to
#  (1 + MAX_REACT_STEPS) search calls. Keep the cap low (2) to stay in the free tier.
# =============================================================================
REACT_RESEARCH = True
MAX_REACT_STEPS = 2


class GlassBoxBot(ForecastBot):
    """Your forecaster, ported to run live on Metaculus."""

    # forecasting-tools spins up many coroutines; keep a gentle concurrency cap so
    # the ensemble passes don't trip provider rate limits.
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build the diverse forecaster pool once, and a cycle to rotate through it.
        self._forecaster_pool = [
            GeneralLlm(model=m, temperature=0.3, timeout=60, allowed_tries=2)
            for m in FORECASTER_POOL
        ]
        self._forecaster_cycle = itertools.cycle(self._forecaster_pool)

    def _next_forecaster_llm(self):
        # UPGRADE 2: hand each pass the next model family in the rotation, so the
        # 3 passes are 3 different minds. The model is picked synchronously at the
        # top of each pass (before any await), so the concurrent passes each take a
        # distinct model. Toggle off -> everyone uses the single "default".
        if DIVERSE_ENSEMBLE:
            return next(self._forecaster_cycle)
        return self.get_llm("default", "llm")

    # --- ReAct research helpers (UPGRADE 3) -------------------------------
    async def _run_search(self, prompt: str) -> str:
        # One search, via AskNews if configured else the researcher LLM. Reused by
        # both the initial pass and every gap-filling follow-up.
        researcher = self.get_llm("researcher")
        if isinstance(researcher, str) and researcher.startswith("asknews"):
            return await AskNewsSearcher().call_preconfigured_version(researcher, prompt)
        return await self.get_llm("researcher", "llm").invoke(prompt)

    async def _identify_gap(self, question, research_so_far: str) -> str | None:
        # Ask a CHEAP model what one thing is still missing. Returns a short search
        # query, or None if the research is already sufficient (the early-stop).
        prompt = clean_indents(
            f"""
            You are refining research for a forecast. Below is the question and the research so far.
            Identify the SINGLE most important piece of information still MISSING that could change
            the forecast. If the research is already sufficient, reply with exactly: DONE
            Otherwise reply with ONLY a short search query (a few words) for the missing piece.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}

            Research so far:
            {research_so_far}
            """
        )
        resp = (await self.get_llm("parser", "llm").invoke(prompt)).strip()
        if not resp or "DONE" in resp.upper()[:8]:
            return None
        return resp[:120]  # keep the query short

    async def _react_expand(self, question, research_so_far: str) -> str:
        # The loop: up to MAX_REACT_STEPS follow-up searches, stopping early the
        # moment the model says the research is sufficient. The range() is the HARD
        # cap — it can never run more than MAX_REACT_STEPS searches, whatever happens.
        if not REACT_RESEARCH:
            return research_so_far
        for step in range(MAX_REACT_STEPS):
            gap = await self._identify_gap(question, research_so_far)
            if not gap:
                break  # model says we have enough — stop early
            more = await self._run_search(
                f"Regarding: {question.question_text}\nFind specifically: {gap}"
            )
            research_so_far += clean_indents(
                f"""

                --- FOLLOW-UP SEARCH (step {step + 1}/{MAX_REACT_STEPS}, filling gap: {gap}) ---
                {more}
                """
            )
        return research_so_far

    # ---------------------------------------------------------------------
    # 1. RESEARCH  +  OPERATIONALISE-AND-BIND   (run once per question, shared)
    # ---------------------------------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # --- (a) the news pass: AskNews if configured, else the researcher LLM
            news_prompt = clean_indents(
                f"""
                You are a research assistant to a superforecaster. Do NOT forecast.
                Give a concise but detailed rundown of the most relevant, recent facts,
                and note which outcome it would currently resolve to.

                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                {question.fine_print}
                """
            )
            news = await self._run_search(news_prompt)
            news = await self._react_expand(question, news)   # UPGRADE 3: gap-filling research loop

            # --- (b) operationalise-and-bind: the crown jewel, ported.
            # For a MULTIPLE-CHOICE question, write the exact real-world condition
            # that resolves EACH official option, straight from the criteria. This
            # block is appended to the research so all ensemble passes share it and
            # stop improvising what "Open"/"US-only"/"Closed" mean.
            binding = ""
            if isinstance(question, MultipleChoiceQuestion):
                rubric_prompt = clean_indents(
                    f"""
                    Below is a forecasting question, its official options, and its resolution criteria.
                    For EACH option, state the precise real-world condition that makes it resolve —
                    binding any ambiguous term (partial/restricted availability, "generally available",
                    etc.) to exactly one option, using the criteria's own definitions. Be terse.

                    Question: {question.question_text}
                    Options: {question.options}
                    Resolution criteria: {question.resolution_criteria}
                    {question.fine_print}

                    Format as one line per option:  <option> => <condition>
                    """
                )
                binding = await self.get_llm("default", "llm").invoke(rubric_prompt)

            research = news + _coverage_recency_note(news, datetime.now())
            research += await _resolution_source_note(question)   # UPGRADE 1: read the actual resolution page
            if binding:
                research += clean_indents(
                    f"""

                    --- BINDING DEFINITIONS (apply these exactly; do not reinterpret an option) ---
                    {binding}
                    """
                )
            logger.info(f"Research for {question.page_url}:\n{research}")
            return research

    # ---------------------------------------------------------------------
    # 2. BINARY FORECAST   (prompt -> reasoning -> structure_output -> float)
    # ---------------------------------------------------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write briefly: (a) time left, (b) the status-quo outcome,
            (c) a scenario giving No, (d) a scenario giving Yes.

            If this is a rate-based "will it happen by then" question, note whether the process is
            self-exciting (clusters — epidemics, volatility, unrest, viral spread) or memoryless, and
            if a COVERAGE RECENCY SIGNAL appears in the research, weigh it as a weak proxy for being
            inside a burst — only for self-exciting processes, never on the proxy alone.

            {TIME_AWARE_PRIOR}

            The last thing you write is: "Probability: ZZ%", 0-100.
            """
        )
        reasoning = await self._next_forecaster_llm().invoke(prompt)  # UPGRADE 2: rotate model family per pass
        parsed: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        p = max(0.01, min(0.99, parsed.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=p, reasoning=reasoning)

    # ---------------------------------------------------------------------
    # 3. MULTIPLE-CHOICE FORECAST   (+ the consistency-check diagnostic)
    # ---------------------------------------------------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question: {question.question_text}
            The options are: {question.options}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            {question.fine_print}

            Your research assistant says (INCLUDING binding definitions — use them exactly):
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write briefly: (a) time left, (b) the status-quo outcome,
            (c) a scenario giving a surprising outcome.

            {TIME_AWARE_PRIOR}
            Also leave moderate probability on most options — do not put ~0 on any plausible option.

            The last thing you write is your final probabilities for the {len(question.options)} options
            in this order {question.options} as:
            Option_A: Probability_A
            ...
            """
        )
        reasoning = await self._next_forecaster_llm().invoke(prompt)  # UPGRADE 2: rotate model family per pass

        parsing_instructions = clean_indents(
            f"""
            Make sure every option name is exactly one of: {question.options}
            Strip any leading "Option" label. Include a 0% option rather than skipping it.
            If the reasoning revised itself, use the LAST set of numbers.
            """
        )
        predicted: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )

        # --- the consistency check (your "Forecaster 1" catch, automated).
        # Ask which option the NARRATIVE implies, compare to the emitted top option,
        # and log a flag if they disagree. We log rather than auto-correct: the
        # ensemble + median already dilute a single rogue pass, and the binding step
        # removes the usual cause. The flag surfaces in the bot's comment for review.
        await self._log_consistency(question, reasoning, predicted)

        return ReasonedPrediction(prediction_value=predicted, reasoning=reasoning)

    async def _log_consistency(self, question, reasoning, predicted: PredictedOptionList):
        try:
            top_emitted = max(predicted.predicted_options, key=lambda o: o.probability).option_name
            check = await self.get_llm("parser", "llm").invoke(clean_indents(
                f"""
                Read this forecaster's reasoning and IGNORE its numbers. Based on the
                narrative alone, which single option does it conclude is most likely?
                Options: {question.options}
                Reply with the option text only.

                Reasoning:
                {reasoning}
                """
            ))
            narrative_top = check.strip().strip('"')
            norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
            if narrative_top and norm(narrative_top) != norm(top_emitted):
                logger.warning(
                    f"[CONSISTENCY] {question.page_url}: numbers favour '{top_emitted}' "
                    f"but the narrative concludes '{narrative_top}'. Words-vs-numbers slip."
                )
        except Exception as e:  # a diagnostic must never break a forecast
            logger.info(f"consistency check skipped: {e}")

    # ---------------------------------------------------------------------
    # 4. NUMERIC  — left as a minimal pass-through for now (MiniBench/seasonal
    #    include numeric questions; build this out when you tackle the "spread
    #    set by the deadline" type. For now delegate to a simple prompt.)
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # 4. NUMERIC  — the base-rate pipeline, made explicit and clustering-aware
    #    A numeric question wants a whole DISTRIBUTION (percentiles), not one
    #    number. The prompt forces the pipeline you drilled by hand:
    #      classify (count vs level) -> build the rate -> CHECK for clustering
    #      -> let the deadline set the SPREAD (sigma grows ~ sqrt(time)).
    #    The Poisson maths (rate x exposure, 1 - e^-lambda) is a SANITY RAIL in
    #    the reasoning, not hard-coded — so the model can and should deviate when
    #    events cluster. Baking the steady-rate formula into code would throw that
    #    away; keeping it in the prompt keeps the clustering judgement alive.
    # ---------------------------------------------------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower_msg, upper_msg = self._bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster producing a probability DISTRIBUTION, not a point estimate.

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Units for the answer: {question.unit_of_measure if question.unit_of_measure else "not stated — infer and state them"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_msg}
            {upper_msg}

            Reason in this exact order:

            STEP 1 — Classify. Is this a COUNT / occurrence question ("how many X by date T",
            "how many times will Y happen") or a LEVEL question (a price, rate, index, measurement)?
            State which, because it changes how you build the estimate.

            STEP 2a — If COUNT, build the base rate explicitly:
              - Name the reference class and how many events it saw over how long, i.e. rate = events / period.
              - State the exposure window until resolution.
              - CLUSTERING CHECK (do not skip). First CLASSIFY the process:
                SELF-EXCITING (contagion / momentum / feedback — epidemics, market volatility, violence
                and conflict, viral or social spread, product launches, funding or M&A waves) or
                MEMORYLESS (independent mechanical failures across a fleet, background accident rates,
                a single volcano, decay). Only self-exciting processes cluster.
                If self-exciting: consult the COVERAGE RECENCY SIGNAL in the research above (a weak,
                retrieval-biased proxy) — a pronounced recent skew is weak evidence you are INSIDE a
                burst, so lift the near-term rate above the long-run base rate, but never on the proxy
                alone. If memoryless: trust the steady base rate and ignore recent coverage density.
                State which class it is, and which way (if at all) you adjusted the rate.
              - SANITY RAIL only: note the steady-rate expectation lambda = rate x exposure, and that
                P(at least one) = 1 - e^(-lambda). Use it to check your head — it is NOT the final answer,
                and it is wrong when events cluster.

            STEP 2b — If LEVEL, anchor on the recent historical distribution and the current trend:
            a central value, and how far the quantity has plausibly moved over windows of similar length.

            STEP 3 — Let the DEADLINE set the SPREAD. Uncertainty grows with time — roughly with the
            square root of the horizon. The further away resolution is, the WIDER your 10-to-90 interval
            must be. A distant question paired with a narrow interval is over-confident.

            STEP 4 — Prior: weight the status quo by default, UNLESS a scheduled event or hard deadline
            inside the window is likely to move the outcome, in which case weight the trajectory.

            STEP 5 — Be humble: set wide 90/10 intervals to survive unknown unknowns. Respect the bounds above.

            Before your answer, write briefly: (a) time left, (b) the status-quo outcome,
            (c) the trend outcome, (d) a low-tail scenario, (e) a high-tail scenario.

            The last thing you write is your final answer, values INCREASING, in the stated units,
            with no scientific notation:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    # bound messages — respect open/closed bounds and nominal bounds (copied from
    # the template's logic so this bot stays self-contained).
    def _bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        u = question.unit_of_measure or ""
        upper_msg = (
            f"The question creator thinks the number is likely not higher than {upper} {u}."
            if question.open_upper_bound else f"The outcome cannot be higher than {upper} {u}."
        )
        lower_msg = (
            f"The question creator thinks the number is likely not lower than {lower} {u}."
            if question.open_lower_bound else f"The outcome cannot be lower than {lower} {u}."
        )
        return lower_msg, upper_msg

    # reasoning -> percentiles -> NumericDistribution  (framework parse + assembly)
    async def _numeric_prompt_to_forecast(
        self, question: NumericQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._next_forecaster_llm().invoke(prompt)  # UPGRADE 2: rotate model family per pass
        parsing_instructions = clean_indents(
            f"""
            The text is a forecast distribution for the numeric question: "{question.question_text}".
            Give each percentile value in the correct units: {question.unit_of_measure}.
            Convert any scientific notation to plain numbers. Values must increase with percentile.
            """
        )
        percentiles: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        prediction = NumericDistribution.from_question(percentiles, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ---------------------------------------------------------------------
    # 5. AGGREGATION OVERRIDE: median for multiple choice
    #    (binary is already median in the framework — this only changes MC).
    #    Why: mean dilutes a correct minority on high-disagreement questions
    #    (the No-Stream post-mortem). Median keeps the consensus, drops the outlier.
    # ---------------------------------------------------------------------
    async def _aggregate_predictions(
        self, predictions: list[PredictionTypes], question: MetaculusQuestion
    ) -> PredictionTypes:
        if predictions and isinstance(predictions[0], PredictedOptionList):
            names = [o.option_name for o in predictions[0].predicted_options]
            merged = []
            for name in names:
                probs = [
                    o.probability
                    for pl in predictions
                    for o in pl.predicted_options
                    if o.option_name == name
                ]
                merged.append((name, statistics.median(probs)))
            total = sum(p for _, p in merged) or 1.0          # renormalise to sum 1
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=n, probability=p / total) for n, p in merged
                ]
            )
        # binary / numeric / everything else -> framework default (already median for binary)
        return await super()._aggregate_predictions(predictions, question)


# =============================================================================
#  RUN SCRIPT
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["tournament", "test_questions"],
        default="test_questions",   # default to the UNSCORED arena — safe for first runs
    )
    mode = ap.parse_args().mode

    bot = GlassBoxBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,   # LEAN: 3 passes, not 5 (cost)
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        # Pin models here. Use OpenRouter (Metaculus donated credits) + AskNews.
        # Spend budget on a strong DEFAULT model; use a cheap one for parsing.
        llms={
            "default": GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.6",  # the score lever
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
            "parser": "openrouter/anthropic/claude-haiku-4.5",    # cheap, for structure_output + consistency
            "researcher": "asknews/news-summaries",               # free search via Metaculus/AskNews
            "summarizer": "openrouter/anthropic/claude-haiku-4.5",
        },
    )

    client = MetaculusClient()
    if mode == "test_questions":
        # The unscored bot-testing arena — run here first to confirm it forecasts
        # and posts a comment without bugs. Turn OFF skip so you can re-run freely
        # while shaking out bugs; nothing here is scored.
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )
    else:
        # Live: forecast open questions in the seasonal tournament AND MiniBench.
        # From here on: NO peeking-then-tweaking on these questions (the rules).
        reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
        reports += asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )

    for r in reports:
        if isinstance(r, Exception):
            logger.error(f"question errored: {r}")
    logger.info(f"done — {len(reports)} questions handled")
