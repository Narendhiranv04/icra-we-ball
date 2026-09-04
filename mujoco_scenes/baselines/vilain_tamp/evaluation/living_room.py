"""Hidden terminal requirements for the living-room benchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import (
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    check,
    physical_on,
    required_sequence,
    required_string,
)


def evaluate_living_room_requirements(
    terminal_state: TerminalStateSnapshot,
    effect_ledger: Sequence[Mapping[str, Any]],
    hidden_context: HiddenBenchmarkContext,
) -> tuple[Mapping[str, Any], ...]:
    """Check the five actual payload/support relations from physical state."""
    del effect_ledger
    requirements = hidden_context.requirements
    left_payloads = required_sequence(requirements, "left_payloads")
    right_payloads = required_sequence(requirements, "right_payloads")
    remote = required_string(requirements, "remote")
    left_support = required_string(requirements, "left_support")
    right_support = required_string(requirements, "right_support")
    shared_support = required_string(requirements, "shared_support")
    required_payloads = (*left_payloads, *right_payloads, remote)
    required_supports = (left_support, right_support, shared_support)

    payloads_present = all(
        terminal_state.objects.get(payload, {}).get("present", True) is True
        and payload in terminal_state.objects
        for payload in required_payloads
    )
    supports_present = all(
        terminal_state.objects.get(support, {}).get("present", False) is True
        for support in required_supports
    )
    left_correct = all(
        physical_on(terminal_state.objects.get(payload, {}), left_support)
        for payload in left_payloads
    )
    right_correct = all(
        physical_on(terminal_state.objects.get(payload, {}), right_support)
        for payload in right_payloads
    )
    remote_correct = physical_on(
        terminal_state.objects.get(remote, {}), shared_support
    )

    return (
        check("required_payloads_present", payloads_present),
        check("required_supports_present", supports_present),
        check("left_cup_and_saucer_physically_on_left_table", left_correct),
        check("right_cup_and_saucer_physically_on_right_table", right_correct),
        check("remote_physically_on_shared_table", remote_correct),
        check("no_payload_held", not terminal_state.held_objects),
    )
