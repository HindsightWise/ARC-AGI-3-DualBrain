"""Small parent-owned coverage ledger; issued attempts are not inferred successes."""
import numpy as np


class ActionCoverage:
    def __init__(self):
        self.actions = set()
        self.click_colours = set()
        self.unknown_click_colour = False

    def issued(self, action, board, *, board_known=True):
        self.actions.add(action.label)
        if action.label != "CLICK":
            return
        row, col = action.row, action.col
        if (board_known and row is not None and col is not None
                and 0 <= row < board.shape[0] and 0 <= col < board.shape[1]):
            # Read the actual pre-action pixel, never the model's claimed colour.
            self.click_colours.add(int(board[row, col]))
        else:
            self.unknown_click_colour = True

    def text(self, available_actions, board, colour_names, rects=()):
        untried = [label for label in available_actions
                   if label not in {"RESET", "UNDO"} and label not in self.actions]
        text = "Untried on this level: " + (", ".join(untried) or "none") + "."
        if "CLICK" in available_actions:
            if self.unknown_click_colour:
                text += " Never-clicked colours: unknown (a click followed an unobserved step)."
            else:
                visible = np.ones(board.shape, dtype=bool)
                for r0, r1, c0, c1 in rects:
                    visible[r0:r1, c0:c1] = False
                colours = sorted(set(map(int, np.unique(board[visible]))) - self.click_colours)
                names = [f"{colour_names.get(c, 'colour')} ({c})" for c in colours]
                scope = "visible colours outside marked region" if rects else "visible colours"
                text += f" Never clicked on this level ({scope}): " + (", ".join(names) or "none") + "."
        return text
