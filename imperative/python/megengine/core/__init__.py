# -*- coding: utf-8 -*-
import os
import sys
from contextlib import contextmanager

from ._imperative_rt.core2 import get_option, set_option
from .tensor.megbrain_graph import Graph
