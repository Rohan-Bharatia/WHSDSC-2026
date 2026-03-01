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

import numpy as np

class SeasonSimulator:
    def __init__(self, poisson_model, elo_system):
        self.poisson = poisson_model
        self.elo = elo_system

    def simulate_game(self, team_a, team_b, features_a, features_b):
        lambda_a = self.poisson.predict(features_a)[0]
        lambda_b = self.poisson.predict(features_b)[0]

        goals_a = np.random.poisson(lambda_a)
        goals_b = np.random.poisson(lambda_b)

        if goals_a == goals_b:
            if np.random.rand() > 0.5:
                goals_a += 1
            else:
                goals_b += 1

        self.elo.update(team_a, team_b, goals_a, goals_b)

        return goals_a, goals_b
