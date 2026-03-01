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
    def __init__(self, model_home, model_away, elo_system):
        self.model_home = model_home
        self.model_away = model_away
        self.elo = elo_system
        self.ot_time_ratio = 5 / 60

    def simulate_overtime(self, lambda_a, lambda_b):
        ot_lambda_a = lambda_a * self.ot_time_ratio
        ot_lambda_b = lambda_b * self.ot_time_ratio

        goals_a = np.random.poisson(ot_lambda_a)
        goals_b = np.random.poisson(ot_lambda_b)

        return goals_a, goals_b

    def simulate_game(self, team_a, team_b, features_a, features_b):
        lambda_a = self.model_home.predict(features_a)[0]
        lambda_b = self.model_away.predict(features_b)[0]

        goals_a = np.random.poisson(lambda_a)
        goals_b = np.random.poisson(lambda_b)

        went_ot = False

        if goals_a == goals_b:
            went_ot = True

            ot_a, ot_b = self.simulate_overtime(lambda_a, lambda_b)

            goals_a += ot_a
            goals_b += ot_b

            if goals_a == goals_b:
                if np.random.rand() < 0.5:
                    goals_a += 1
                else:
                    goals_b += 1

        self.elo.update(team_a, team_b, goals_a, goals_b)

        return goals_a, goals_b, went_ot
