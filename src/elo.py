# MIT License
#
# Copyright (c) 2026 Rohan Bharatia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from collections import defaultdict

K = 20
BASE_ELO = 1500

class EloSystem:
    def __init__(self):
        self.ratings = defaultdict(lambda: BASE_ELO)

    def expected(self, team_a, team_b):
        ra = self.ratings[team_a]
        rb = self.ratings[team_b]
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, team_a, team_b, score_a, score_b):
        exp_a = self.expected(team_a, team_b)
        result_a = 1 if score_a > score_b else 0

        self.ratings[team_a] += K * (result_a - exp_a)
        self.ratings[team_b] += K * ((1 - result_a) - (1 - exp_a))
