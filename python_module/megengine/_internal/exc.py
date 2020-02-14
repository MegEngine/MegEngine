# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""exception handling"""

from . import mgb as _mgb


class MegBrainError(Exception):
    """exception class used by megbrain library"""

    tracker = None
    """the tracker setup by :func:`.set_exc_opr_tracker` when the related
    operator is created"""

    tracker_grad_orig = None
    """if this operator is created by taking gradient, this var would be the
    tracker of the operator that causes the grad."""

    def __init__(self, msg, tracker, tracker_grad_orig):
        assert isinstance(msg, str)
        super().__init__(msg, tracker, tracker_grad_orig)
        self.tracker = tracker
        self.tracker_grad_orig = tracker_grad_orig

    @classmethod
    def _format_tracker(cls, tracker):
        return ("| " + i for i in str(tracker).split("\n"))

    def __str__(self):
        lines = []
        lines.extend(self.args[0].split("\n"))
        if self.tracker is not None:
            lines.append("Exception tracker:")
            lines.extend(self._format_tracker(self.tracker))
        if self.tracker_grad_orig is not None:
            lines.append(
                "Exception caused by taking grad of another operator with tracker:"
            )
            lines.extend(self._format_tracker(self.tracker_grad_orig))
        while not lines[-1].strip():
            lines.pop()
        for idx, ct in enumerate(lines):
            if ct.startswith("bt:"):
                lines[idx] = "+ " + lines[idx]
                for t in range(idx + 1, len(lines)):
                    lines[t] = "| " + lines[t]
                break
        return "\n".join(lines)


_mgb._reg_exception_class(MegBrainError)
