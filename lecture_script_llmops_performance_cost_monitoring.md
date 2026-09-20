# Lecture Script: LLMOps — Performance, Cost, and Monitoring
**Format:** Facilitator-facing live script | **Duration:** 110 minutes | **Level:** Beginner

---

## Session Flow at a Glance

| Block | Topic | Time |
|---|---|---|
| 1 | Why Does This Matter? | 8 min |
| 2 | Caching Strategies in Production | 20 min |
| 3 | Latency Reduction | 22 min |
| 4 | Cost Budgeting | 22 min |
| 5 | Monitoring | 28 min |
| 6 | Lecture Summary and Recap | 10 min |

---

## Block 1 — Why Does This Matter?

> 🎯 **Instructor Note:** This audience already covered prompt versioning, evaluation sets, and regression testing in the previous LLMOps session — the discipline of ensuring a prompt change does not silently break correctness. This session covers the second half of running an LLM-powered feature responsibly in production: not "is it still correct," but "is it still fast enough, affordable enough, and visible enough that you would actually know if something went wrong." Open by naming that distinct half explicitly. Wait after the opening question.

**[Script:]**

"The last session answered a specific question: how do you know a prompt change did not break correctness? Today answers a different set of questions that matter just as much once a feature is actually live: is it responding fast enough that users do not give up waiting? Is it staying within a cost you can actually afford as usage grows? And critically — if something does go wrong, how would you actually find out, and how quickly?

These are not new problems invented for LLM features specifically — you already covered caching and cost fundamentals in the AI cost optimization session. Today builds on that foundation with a production-operations lens: not just 'here is a technique that reduces cost,' but 'here is how you actually run this technique reliably, at real scale, and know whether it is working.' Caching gets revisited specifically for how you monitor and validate it in production, not just how it works. Latency reduction covers techniques for making a response feel and actually be fast, which matters independently of pure cost. Cost budgeting covers setting real limits and getting alerted before you exceed them, not just calculating cost after the fact. And monitoring covers the piece that ties everything in both LLMOps sessions together: the actual dashboards, alerts, and visibility that let you know, in real time, whether your LLM-powered feature is healthy — correct, fast, and affordable — rather than finding out from an angry user or a surprising bill."

---

## Block 2 — Caching Strategies in Production

### 2A — A Quick Recap, and What This Session Adds

**[Script:]**

"You already know the core caching mechanism from the cost optimization session — hash the input, check for a stored response, avoid a redundant API call on a match. That mechanism does not change today. What we add here is the production-operations layer around it: how do you know your cache is actually working, how do you decide what belongs in it, and what happens when the underlying data it is caching changes."

---

### 2B — Tracking Cache Hit Rate

**[Script:]**

"Cache hit rate — the percentage of requests served from cache rather than requiring a real API call — is the single most important number for understanding whether your caching strategy is actually earning its complexity. A cache with a five percent hit rate is barely helping at all, and its cost and maintenance overhead may not be worth it. A cache with a seventy percent hit rate is meaningfully reducing both cost and latency for the majority of your traffic."

> 🎯 **Instructor Note:** Ask: "If you deployed a cache and never measured its hit rate, how would you know whether it was actually worth the added complexity in your codebase?" Answer: you would not know — you would only have an assumption that caching helps, without any real evidence it is working as intended for your actual traffic patterns, which might look very different from what you initially expected.

**Demo 1 — Tracking cache hit rate (whiteboard-friendly)**

```python
cache_stats = {"hits": 0, "misses": 0}

def get_cached_ai_response(prompt: str, system_prompt: str) -> str:
    cache_key = hashlib.sha256((system_prompt + prompt).encode()).hexdigest()

    if cache_key in response_cache:
        cache_stats["hits"] += 1
        return response_cache[cache_key]

    cache_stats["misses"] += 1
    result = get_ai_response(prompt, system_prompt)
    response_cache[cache_key] = result
    return result

def get_hit_rate() -> float:
    total = cache_stats["hits"] + cache_stats["misses"]
    return cache_stats["hits"] / total if total > 0 else 0.0
```

**[Script:]**

"This is the exact same caching function from the cost optimization session, with two counters added — `hits` and `misses` — incremented at exactly the point each outcome occurs. `get_hit_rate()` gives you a real, measurable number, one you would log and monitor over time, not just assume. A hit rate that starts high and gradually declines over weeks is a genuine signal worth investigating — perhaps user queries are becoming more varied, or perhaps the cache needs a different invalidation strategy, covered next."

---

### 2C — Cache Invalidation in Production

**Predict before running: What will happen?**

> 🎯 **Instructor Note:** Ask: "If your application caches responses about your product catalog, and the catalog changes — a product is discontinued — what happens to a cached response that still describes the discontinued product as available, if nothing ever invalidates it?" Answer: the cache would keep confidently serving outdated, now-incorrect information indefinitely, since nothing in the simple caching mechanism from before knows the underlying source data changed. This is the exact same staleness risk named briefly in the RAG session's knowledge base discussion, now applied specifically to response caching.

**[Script:]**

"A cache is only as trustworthy as its invalidation strategy — the mechanism for removing or refreshing entries that are no longer accurate. Three common approaches, each fitting different situations."

> 🎯 **Instructor Note:** Write these on the board.

```
Cache invalidation strategies:
- Time-based (TTL) — entries automatically expire after a set 
  duration, regardless of whether the underlying data actually changed
- Event-based — explicitly clear or update specific cache entries 
  when the underlying data changes, e.g. when a product is updated
- Manual — an operator explicitly clears the cache when needed, 
  suitable only for low-stakes or rarely-changing content
```

**Demo 2 — Time-based cache expiration (whiteboard-friendly)**

```python
import time

def get_cached_ai_response(prompt: str, system_prompt: str, ttl_seconds: int = 3600) -> str:
    cache_key = hashlib.sha256((system_prompt + prompt).encode()).hexdigest()
    cached = response_cache.get(cache_key)

    if cached and (time.time() - cached["timestamp"]) < ttl_seconds:
        cache_stats["hits"] += 1
        return cached["value"]

    cache_stats["misses"] += 1
    result = get_ai_response(prompt, system_prompt)
    response_cache[cache_key] = {"value": result, "timestamp": time.time()}
    return result
```

**[Script:]**

"Each cache entry now stores a timestamp alongside its value. On lookup, an entry is only considered a valid hit if it is younger than `ttl_seconds` — time to live. This guarantees no cached response is ever served indefinitely, no matter what; it will refresh automatically after the specified duration, bounding how stale a cached answer can possibly become, even without any explicit event triggering a refresh."

> 🎯 **Instructor Note:** Ask: "How would you decide what TTL value is actually appropriate for a given cached response — is there one universally correct number?" Answer: no — it depends entirely on how frequently the underlying information actually changes and how costly serving a stale answer would be; a cached answer about a stable company policy might reasonably use a TTL of days, while anything referencing current, frequently-changing data needs a much shorter TTL, or event-based invalidation instead.

**Recap of Block 2 before moving on:**

- Cache hit rate is the key metric for whether a caching strategy is actually earning its complexity, and should be measured, not assumed
- Cache staleness is a genuine production risk, exactly like knowledge base staleness in a RAG system — a cache with no invalidation strategy can confidently serve outdated information indefinitely
- Time-based expiration (TTL) bounds staleness automatically; event-based invalidation is more precise but requires explicitly wiring cache updates to underlying data changes
- The right TTL or invalidation strategy depends on how frequently the underlying information changes and the real cost of serving a stale answer

---

## Block 3 — Latency Reduction

### 3A — Why Latency Matters Independently of Cost

**[Script:]**

"Cost and latency are related but genuinely separate concerns — a request can be perfectly affordable and still feel unacceptably slow to a real user waiting on a response. Latency specifically shapes user experience in a way raw cost does not: a user does not see your bill, but they absolutely notice a three-second delay before anything appears on screen."

---

### 3B — Streaming Responses

**[Script:]**

"One of the most effective latency-reduction techniques does not actually make the model generate faster at all — it changes when the user starts seeing output. Streaming sends each piece of the generated response to the user as it is produced, rather than waiting for the entire response to finish generating before sending anything."

> 🎯 **Instructor Note:** Draw this contrast on the board.

```
Without streaming:
  User sends request → [wait for FULL response to generate] → 
  entire response appears at once
  Perceived latency = full generation time

With streaming:
  User sends request → first tokens appear almost immediately → 
  rest of the response continues appearing as it generates
  Perceived latency = time to FIRST token, which is much shorter
```

**Predict before running: What will happen?**

> 🎯 **Instructor Note:** Ask: "If a full response takes four seconds to generate completely, but the first word appears after 300 milliseconds with streaming enabled, does the user's actual wait time for the full response change at all?" Answer: no — the total time to receive the complete response is unchanged. What changes is the user's perceived experience: they see something happening almost immediately, rather than staring at a blank screen for four full seconds, which meaningfully changes how the wait actually feels even though the underlying compute time is identical.

**Demo 3 — Enabling streaming in an API request (whiteboard-friendly)**

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**[Script:]**

"`stream=True` changes the API's behavior from returning one complete response object to returning a sequence of small chunks as they are generated. The loop processes each chunk as it arrives, printing it immediately rather than waiting to accumulate the full response first. In a real FastAPI application, this same pattern streams chunks directly to the client as a streaming HTTP response, so the user's interface can display text appearing progressively, exactly like watching a response being typed in real time."

> 🎯 **Instructor Note:** Ask: "For which kinds of features does streaming provide the most real user-experience benefit — a long-form chat response, or a short classification that returns a single word like 'positive' or 'negative'?" Answer: streaming provides much more benefit for longer responses, where the gap between "nothing visible yet" and "fully complete" is large. For a single-word classification, the entire response might generate faster than a human even perceives a delay, so streaming adds negligible practical benefit there.

---

### 3C — Reducing Latency Through Model and Prompt Choices

**[Script:]**

"Beyond streaming, several choices from earlier sessions directly affect latency, not just cost. Model selection matters here too — a smaller model, chosen deliberately for a task that does not need a large model's full capability, is genuinely faster to generate a response from, not just cheaper per token. Prompt optimization reduces latency for the same reason it reduces cost — fewer input tokens to process means less time spent before generation even begins. And caching, from Block 2, is in some sense the ultimate latency reduction: a cache hit has effectively zero generation latency at all."

> 🎯 **Instructor Note:** Ask a synthesis question connecting back to earlier sessions: "Which specific technique from earlier LLMOps and cost optimization content directly serves both cost reduction and latency reduction simultaneously, rather than trading one for the other?" Answer: caching — a cache hit avoids both the token cost and the generation time of a real API call entirely, making it one of the few techniques that improves both dimensions at once rather than requiring a tradeoff between them.

**Recap of Block 3 before moving on:**

- Latency and cost are related but distinct concerns; a request can be affordable and still feel unacceptably slow
- Streaming does not reduce total generation time, but dramatically reduces perceived latency by showing output as it is produced rather than all at once
- Streaming provides the most benefit for longer responses; negligible benefit for very short ones
- Model selection, prompt optimization, and caching all reduce latency as well as cost, with caching improving both simultaneously

---

## Block 4 — Cost Budgeting

### 4A — From Calculating Cost to Actively Budgeting It

**[Script:]**

"The cost optimization session covered calculating cost after a request completes — `response.usage`, logged and aggregated. Budgeting is the more proactive discipline: setting real limits ahead of time, tracking spend against those limits continuously, and getting alerted before a limit is exceeded, rather than only discovering an overage after the bill arrives."

---

### 4B — Setting and Enforcing a Budget

**Predict before running: What will happen?**

> 🎯 **Instructor Note:** Ask: "If your application has no budget enforcement at all, and a bug causes a feature to call the API in an unintended infinite loop overnight, what would actually stop that from happening, and when would you likely find out?" Answer: nothing would stop it automatically, and you would likely only find out the next morning, or whenever someone happens to check the bill — this is precisely the runaway-cost risk named in the agentic debugging session's `max_iterations` discussion, now viewed at the level of an entire application's total spend rather than a single agent run.

**Demo 4 — Enforcing a daily spending budget (whiteboard-friendly)**

```python
DAILY_BUDGET = 50.00  # dollars
daily_spend = {"amount": 0.0, "date": None}

def check_and_record_spend(cost: float):
    today = datetime.date.today()
    if daily_spend["date"] != today:
        daily_spend["date"] = today
        daily_spend["amount"] = 0.0

    if daily_spend["amount"] + cost > DAILY_BUDGET:
        raise BudgetExceededError(f"Daily budget of ${DAILY_BUDGET} would be exceeded")

    daily_spend["amount"] += cost
```

**[Script:]**

"`check_and_record_spend` resets the tracked amount at the start of each new day, and raises an explicit error before a request that would push total spend past the daily budget is even allowed to proceed. This is a hard stop, not just a warning — it actively prevents runaway spend rather than only reporting it after the fact.

In a real production system, `BudgetExceededError` might trigger a graceful fallback — a cached or simpler response, a clear message to the user that the service is temporarily limited — rather than the request simply failing outright with an unhandled error, exactly the graceful-failure discipline from the agentic systems and debugging sessions applied here at the budget level."

> 🎯 **Instructor Note:** Ask: "Is a hard budget cutoff always the right response when a budget is nearly exceeded, or might a different response be more appropriate depending on the feature?" Answer: it depends on the feature's importance — a hard cutoff might be entirely appropriate for a non-critical, nice-to-have feature, while a mission-critical feature might instead warrant an alert to the team and a temporary fallback to a cheaper model, rather than fully denying service to users. Budget enforcement is itself a design decision, not a one-size-fits-all rule.

---

### 4C — Budgeting at Multiple Levels

**[Script:]**

"A single global budget is a reasonable starting point, but real systems often benefit from budgeting at multiple levels — per feature, so one runaway feature cannot silently consume the budget intended for everything else; per user or per customer, particularly relevant if different customers are on different pricing tiers; and per time window, both daily and monthly, since a daily budget alone does not protect against a sustained, moderate overage accumulating unnoticed across an entire month."

> 🎯 **Instructor Note:** Ask: "Why might per-feature budgeting matter even if your total daily budget is never actually exceeded?" Answer: a single global budget can hide an important underlying problem — one feature could be consuming the vast majority of the budget while contributing relatively little value, silently starving other features of the room to operate normally, even though the top-line total number looks perfectly fine.

**Recap of Block 4 before moving on:**

- Budgeting is proactive — setting real limits and tracking spend against them continuously — rather than only calculating cost after the fact
- A hard budget enforcement mechanism actively prevents runaway spend, directly analogous to the `max_iterations` safeguard from agentic systems, applied at the level of total application spend
- The right response to an approaching budget limit depends on the feature's importance — a hard cutoff, a graceful fallback, or an alert are all legitimate options depending on context
- Real systems often benefit from budgeting at multiple levels — per feature, per user, and per time window — since a single global number can hide important underlying imbalances

---

## Block 5 — Monitoring

### 5A — Why Monitoring Ties Everything Together

**[Script:]**

"Everything covered across both LLMOps sessions — prompt versioning, evaluation sets, regression testing, caching, latency, and budgeting — only genuinely protects you in production if you can actually observe it happening in real time. Monitoring is the discipline of making your system's real behavior visible, continuously, rather than only discovering a problem when a user complains or a bill surprises you."

---

### 5B — What to Actually Monitor

**[Script:]**

"Several categories of metrics matter specifically for an LLM-powered feature, building directly on everything from both sessions."

> 🎯 **Instructor Note:** Write this consolidated monitoring checklist on the board — this is the practical takeaway of the entire two-session LLMOps sequence.

```
What to monitor for an LLM-powered feature:
- Cost: total spend, spend per feature, trend over time (Block 4)
- Latency: response time, time-to-first-token for streaming (Block 3)
- Cache performance: hit rate, trend over time (Block 2)
- Quality: evaluation set pass rate over time, per prompt version 
  (previous LLMOps session)
- Errors: API failures, timeout rate, budget-exceeded events
- Usage volume: request count, to contextualize all of the above
```

**[Script:]**

"Notice this checklist is not new material invented for this block — it is the metrics from every earlier block in both LLMOps sessions, brought together into one place you would actually watch continuously, rather than checking manually and occasionally."

---

### 5C — Setting Up Meaningful Alerts

**Predict before running: What will happen?**

> 🎯 **Instructor Note:** Ask: "If you have a dashboard showing all of these metrics, but nobody is actively looking at it at 2 AM when a real problem starts, how would you actually find out something went wrong in time to matter?" Answer: you likely would not, until real damage had already accumulated — a dashboard that requires someone to be actively watching it is fundamentally different from an alert that proactively notifies you the moment a real threshold is crossed, regardless of whether anyone happened to be looking at that exact moment.

**Demo 5 — A simple alerting check (whiteboard-friendly)**

```python
def check_alerts():
    hit_rate = get_hit_rate()
    if hit_rate < 0.20:
        send_alert(f"Cache hit rate dropped to {hit_rate:.1%} — investigate")

    if daily_spend["amount"] > DAILY_BUDGET * 0.80:
        send_alert(f"Daily spend at {daily_spend['amount']:.2f} — 80% of budget reached")

    error_rate = get_recent_error_rate()
    if error_rate > 0.05:
        send_alert(f"API error rate at {error_rate:.1%} — investigate immediately")
```

**[Script:]**

"Each check compares a real, currently-monitored metric against a meaningful threshold, and proactively sends an alert — to a team chat channel, an email, a paging system — the moment that threshold is crossed, rather than requiring a human to notice the problem by actively checking a dashboard. The eighty percent budget threshold specifically is deliberately set below one hundred percent — an early warning while there is still time to investigate and respond, rather than only alerting after the hard budget limit from Block 4 has already been hit and requests are actively failing."

> 🎯 **Instructor Note:** Ask: "Why alert at eighty percent of budget rather than waiting until the budget is fully exceeded, given that Block 4 already has a hard enforcement mechanism at one hundred percent?" Answer: the eighty percent alert is a genuine early warning — it gives a human time to investigate why spend is trending unusually high and potentially intervene, before the hard cutoff actually starts denying real requests to real users. The two mechanisms serve different purposes: one is proactive early warning, the other is a last-resort safety net.

**[Script:]**

"A genuinely mature monitoring setup connects directly back to the regression testing from the previous session as well — evaluation set pass rate, tracked continuously in production and not just at prompt-change time, can reveal a quality regression caused by something other than a deliberate prompt change entirely — a provider silently updating the underlying model, for instance, which is a real and recurring risk worth actively watching for, not just something checked once when a prompt itself changes."

> 🎯 **Instructor Note:** This closing point connects the entire two-session sequence together explicitly. Emphasize: "Monitoring evaluation pass rate continuously, not just when you change a prompt, is how you catch the case where nothing about your own code or prompt changed at all, but the underlying provider updated their model and your feature's quality shifted anyway — a risk entirely outside your own control, and one only continuous monitoring, not point-in-time regression testing alone, can actually catch."

**Recap of Block 5 before moving on:**

- Monitoring makes a system's real behavior continuously visible, tying together every metric covered across both LLMOps sessions
- A comprehensive monitoring checklist includes cost, latency, cache performance, quality via evaluation pass rate, error rates, and usage volume
- Proactive alerts, triggered automatically when a threshold is crossed, are fundamentally more reliable than a dashboard that depends on someone actively watching it
- Continuously monitoring evaluation pass rate in production, not just at prompt-change time, can catch quality regressions caused by factors outside your own changes, such as a silent underlying model update from the provider

---

## Block 6 — Lecture Summary

> 🎯 **Instructor Note:** Deliver as active recall. Ask before confirming. "Why does cache hit rate need to be measured rather than assumed? Why does streaming reduce perceived latency without reducing total generation time? What is the difference between calculating cost after the fact and actively budgeting it? Why is a proactive alert more reliable than a dashboard alone?"

**Caching Strategies in Production**

- Cache hit rate is the key metric determining whether caching is actually earning its complexity, and must be measured, not assumed
- Cache staleness is a genuine production risk without a deliberate invalidation strategy
- Time-based expiration bounds staleness automatically; event-based invalidation is more precise but requires explicit wiring to data changes

**Latency Reduction**

- Latency and cost are distinct concerns; a request can be affordable and still feel unacceptably slow
- Streaming reduces perceived latency dramatically by showing output as it generates, without reducing total generation time
- Model selection, prompt optimization, and caching all reduce latency as well as cost; caching improves both simultaneously

**Cost Budgeting**

- Budgeting is proactive — real limits, tracked continuously — rather than only calculating cost after a request completes
- Hard budget enforcement directly parallels the `max_iterations` safeguard from agentic systems, applied at the level of total application spend
- Budgeting at multiple levels — per feature, per user, per time window — reveals imbalances a single global number can hide

**Monitoring**

- Monitoring ties together every metric from both LLMOps sessions into continuous, real-time visibility
- Proactive alerts triggered by threshold crossings are more reliable than a dashboard requiring active human attention
- Continuous monitoring of evaluation pass rate, not just point-in-time regression testing, catches quality regressions caused by factors outside your own changes, like a silent provider-side model update

**Why All of This Matters Together**

- Caching, latency reduction, cost budgeting, and monitoring are the second half of running an LLM-powered feature responsibly in production, completing what the previous session's correctness-focused practices started — prompt versioning and evaluation sets tell you the feature is still right, and today's practices tell you it is still fast, affordable, and visible enough that you would actually know the moment any of that stopped being true; together, both sessions form the complete operational discipline that turns an LLM-powered feature from something that worked in a demo into something genuinely trustworthy to run at real scale, for real users, over real time

---

*End of script.*
