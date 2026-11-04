"""vLLM generation controls; these never change the rendered prompt prefix."""
import os

THINKING_END = (
    "\nThinking budget reached. Act now. Revalidate assumptions using the latest untried-action "
    "and never-clicked-colour list. If one of these would refute the claim you are about to act on, "
    "do the cheapest such single action in this run. Otherwise act on the evidence you have.\n</think>"
)


def thinking_token_budget(output_cap, *, recovery=False):
    """Reserve space for a complete call, including on a shortened recovery request."""
    name = "V2_RECOVERY_THINKING_TOKENS" if recovery else "V2_THINKING_TOKEN_BUDGET"
    budget = int(os.environ.get(name, "1000" if recovery else "12000"))
    if budget == -1:
        return None
    if budget < 0:
        raise ValueError(f"{name} must be nonnegative, or -1 to disable")
    return min(budget, max(0, output_cap - 512))
