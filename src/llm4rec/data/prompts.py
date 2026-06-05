"""Prompt templates for LLM-based binary recommendation scoring.

Each template converts a user's purchase history and a candidate item into a
natural-language prompt whose expected answer is **Yes** or **No**.

Template registry
-----------------
Templates are keyed by a ``prompt_id`` string.  Add new variants by
registering them in ``PROMPT_TEMPLATES``.
"""

from __future__ import annotations

from typing import Callable, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Template definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each template is a callable:
#   (user_id: int, history: List[str], candidate: str) → str

def _template_2_11(user_id: int, history: List[str], candidate: str) -> str:
    """Prompt 2-11: direct purchase-history formulation."""
    if not history:
        return f"Does the user likely to buy {candidate} next?"
    purchase_history = ", ".join(history)
    return (
        f"User_{user_id} has the following purchase history:\n"
        f"{purchase_history}\n"
        f"Does the user likely to buy {candidate} next?"
    )


def _template_2_12(user_id: int, history: List[str], candidate: str) -> str:
    """Prompt 2-12: third-person user-description formulation."""
    if not history:
        return f"Predict whether the user will purchase {candidate} next?"
    purchase_history = ", ".join(history)
    return (
        f"According to User_{user_id}'s purchase history list:\n"
        f"{purchase_history}\n"
        f"Predict whether the user will purchase {candidate} next?"
    )


def _template_3_1(user_id: int, history: List[str], candidate: str) -> str:
    """Prompt 3-1: numbered history list."""
    if not history:
        return f"Based on the purchase history, will the user buy {candidate} next?"
    numbered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(history))
    return (
        f"Here is the purchase history of User_{user_id}:\n"
        f"{numbered}\n\n"
        f"Based on the purchase history, will the user buy {candidate} next?"
    )


def _template_3_2(user_id: int, history: List[str], candidate: str) -> str:
    """Prompt 3-2: preference-oriented formulation."""
    if not history:
        return f"Is the user likely to purchase {candidate} as their next item?"
    purchase_history = ", ".join(history)
    return (
        f"User_{user_id} has previously purchased: {purchase_history}.\n"
        f"Given these preferences, is the user likely to purchase "
        f"{candidate} as their next item?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Template registry
# ─────────────────────────────────────────────────────────────────────────────

PromptFn = Callable[[int, List[str], str], str]

PROMPT_TEMPLATES: Dict[str, PromptFn] = {
    "2-11": _template_2_11,
    "2-12": _template_2_12,
    "3-1":  _template_3_1,
    "3-2":  _template_3_2,
}

DEFAULT_PROMPT_ID = "2-11"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt(
    prompt_id: str,
    user_id: int,
    history: List[str],
    candidate: str,
) -> str:
    """Render a prompt string for the given template and inputs.

    Parameters
    ----------
    prompt_id : key into ``PROMPT_TEMPLATES``.
    user_id : integer user index.
    history : list of item *names* (strings) the user has purchased.
    candidate : name of the candidate item.

    Returns
    -------
    str : the fully rendered prompt (without the target answer).

    Raises
    ------
    ValueError : if ``prompt_id`` is not registered.
    """
    if prompt_id not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_id={prompt_id!r}. Available: {available}"
        )
    return PROMPT_TEMPLATES[prompt_id](user_id, history, candidate)


def format_target(label: int) -> str:
    """Return the target string for a binary label (1 → 'Yes', 0 → 'No')."""
    return "Yes" if label == 1 else "No"


def list_templates() -> List[str]:
    """Return all registered prompt template IDs."""
    return sorted(PROMPT_TEMPLATES.keys())
