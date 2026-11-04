"""Bounded observational reminders, independent of model-assigned object roles.

Component matches are tentative. Compare the same action/location and either the
same colour/region or an exact reversal of measured shapes. The latter can involve
different components: name both observations, never infer identity or a mechanic.
Resets are reported separately and never become action-result evidence.
"""
from collections import OrderedDict, deque


DIRECTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


def rect_text(rect):
    r0, r1, c0, c1 = rect
    return f"[{r0}:{r1}, {c0}:{c1}]"


def shape_changes(diff, indicator_rects=()):
    """Use existing measured component matches, excluding any indicator overlap."""
    for event in (diff or {}).get("reshaped", []):
        before, after = event["from_rect"], event["to_rect"]
        region = (min(before[0], after[0]), max(before[1], after[1]),
                  min(before[2], after[2]), max(before[3], after[3]))
        if any(region[0] < q[1] and q[0] < region[1] and region[2] < q[3] and q[2] < region[3]
               for q in indicator_rects):
            continue
        yield event, region


def action_key(action):
    parts = action.split()
    # The claimed pixel value is a click validation argument, not a different
    # control. Say "click location" in feedback; this does not identify a trigger.
    return tuple(parts[:3] if parts[0] == "CLICK" else parts[:1])


class ExplorationFeedback:
    MAX_OBSERVATIONS = 128
    MAX_COVERAGE_BOARDS = 256
    MAX_NOTICES = 64
    MAX_EVENTS_PER_TURN = 2

    def __init__(self):
        self.scope = None
        self.observations = deque(maxlen=self.MAX_OBSERVATIONS)
        self.contrasts_shown = deque(maxlen=self.MAX_NOTICES)
        self.coverage = OrderedDict()
        self.shapes_shown = deque(maxlen=self.MAX_NOTICES)
        self.sequence = 0
        self.begin_turn()

    def begin_turn(self):
        self.events_shown = 0

    def restart(self):
        # Preserve factual action outcomes from this level, but a new attempt
        # can have different hidden state. Do not reuse its exploration coverage.
        self.coverage.clear()
        self.shapes_shown.clear()

    def set_scope(self, scope):
        if scope != self.scope:
            self.scope = scope
            self.observations.clear()
            self.contrasts_shown.clear()
            self.restart()

    def observe(self, action, diff, before_key, after_key, available, *, scope, indicator_rects=(), history_index=None):
        self.set_scope(scope)
        trigger = action_key(action)
        if trigger[0] in ("RESET", "UNDO"):
            return []
        self.sequence += 1
        tested = self.coverage.setdefault(before_key, set())
        tested.add(trigger[0])
        self.coverage.move_to_end(before_key)
        if len(self.coverage) > self.MAX_COVERAGE_BOARDS:
            self.coverage.popitem(last=False)

        lines = []
        action_noticed = False
        tally_shown = hint_shown = False
        for event, region in shape_changes(diff, indicator_rects):
            key = (trigger, event["color"], region)
            signature = (event["from_size"], event.get("from_shape_hash"),
                         event["to_size"], event.get("to_shape_hash"))
            observation = (key, signature, history_index, self.sequence)
            self.observations.append(observation)
            prior = next((o for o in reversed(self.observations)
                          if o[3] != self.sequence and o[0] == key and o[1] != signature), None)
            inverse = signature[2:] + signature[:2]
            reversed_shapes = False
            if prior is None and signature != inverse and all((signature[1], signature[3])):
                prior = next((o for o in reversed(self.observations)
                              if o[3] != self.sequence and o[0][0] == trigger and o[1] == inverse), None)
                reversed_shapes = prior is not None
            contrast = ((trigger if reversed_shapes else key), frozenset((prior[1], signature))) if prior else None
            shape = (event["color"], tuple(event["to_rect"]), signature[2:])
            if not action_noticed and self.events_shown >= self.MAX_EVENTS_PER_TURN:
                continue
            notice = []
            if prior and not tally_shown and contrast not in self.contrasts_shown:
                counts = [sum((o[0][0] == trigger if reversed_shapes else o[0] == key) and o[1] == sig
                              for o in self.observations)
                          for sig in (prior[1], signature)]
                def outcome(sig, count):
                    change = f"{sig[0]}->{sig[2]} cells" if sig[0] != sig[2] else f"shape changed ({sig[0]} cells)"
                    return f"{change} x{count}"
                citation = f" (earlier history[{prior[2]}], now history[{history_index}])" if prior[2] is not None and history_index is not None else ""
                control = "click location " + " ".join(trigger[1:]) if trigger[0] == "CLICK" else trigger[0]
                subject = (f"earlier color {prior[0][1]} within {rect_text(prior[0][2])}; "
                           f"now color {event['color']} within {rect_text(region)}" if reversed_shapes else
                           f"color {event['color']} within {rect_text(region)}")
                notice.append(f"Recent same-action observations for {control} ({subject}): "
                              f"{outcome(prior[1], counts[0])}; {outcome(signature, counts[1])}{citation}. "
                              "Component identity and starting states may differ; check the starting state before inferring a rule.")
                self.contrasts_shown.append(contrast)
                tally_shown = True
            if not hint_shown and shape not in self.shapes_shown:
                untried = [d for d in DIRECTIONS if d in available and d not in self.coverage.get(after_key, ())]
                if untried:
                    notice.append("After this shape change, directions with no recorded test from this resulting board "
                                  "(indicator ignored; recent coverage in this attempt): " + ", ".join(untried) + ".")
                    self.shapes_shown.append(shape)
                    hint_shown = True
            if notice:
                if not action_noticed:
                    self.events_shown += 1
                lines.extend(notice)
                # At most one contrast and one direction hint per action. Still
                # retain other events, and let a later component add a contrast
                # even when an earlier component already supplied the hint.
                action_noticed = True
        return lines
