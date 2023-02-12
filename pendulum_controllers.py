from abc import ABC, abstractmethod
from simpful import *


##### Cart Pole Env info:

# ### Action Space

# The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

# | Num | Action | Min  | Max |
# |-----|--------|------|-----|
# | 0   | Torque | -2.0 | 2.0 |


# ### Observation Space

# The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free end and its
# angular velocity.

# | Num | Observation      | Min  | Max |
# |-----|------------------|------|-----|
# | 0   | x = cos(theta)   | -1.0 | 1.0 |
# | 1   | y = sin(angle)   | -1.0 | 1.0 |
# | 2   | Angular Velocity | -8.0 | 8.0 |


# ### Rewards

# The reward function is defined as:
# *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
# where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
# Based on the above equation, the minimum reward that can be obtained is *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> +
# 0.001 * 2<sup>2</sup>) = -16.2736044*, while the maximum reward is zero (pendulum is upright with zero velocity and
# no torque applied).


# ### Starting State

# The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

# ### Episode Termination
# The episode terminates at 200 time steps.


class Controller(ABC):
    def __init__(self, action_space, **kwargs):
        self.action_space = action_space

    @abstractmethod
    def get_action(self, observation):
        pass


class RandomController(Controller):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, observation):
        return self.action_space.sample()


class TestController(Controller):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, observation):
        return [1.3]  # mozna zmieniac wartosci by naocznie zobaczyc wplyw wartosci akcji


class FuzzyController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fis = self.construct_fis()
        self.fis.produce_figure("my_pendulum_controller")
        self.observation_keys = ["Y", "X", "AngV"]
        self.output_keys = ["Torque"]

    def construct_fis(self):
        FS = FuzzySystem(show_banner=False)

        # Zdefiniuj wartości i zmienne lingwistyczne dla wejść do systemu
        o1 = FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.5, d=-0.25), term="BPrawo")
        o2 = FuzzySet(function=Trapezoidal_MF(a=-0.5, b=-0.25, c=-0.05, d=0.0), term="TPrawo")
        o3 = FuzzySet(function=Triangular_MF(a=-0.4, b=0, c=0.4), term="Srodek")
        o4 = FuzzySet(function=Trapezoidal_MF(a=0, b=0.05, c=0.25, d=0.5), term="TLewo")
        o5 = FuzzySet(function=Trapezoidal_MF(a=0.25, b=0.5, c=1, d=1), term="BLewo")
        FS.add_linguistic_variable("X", LinguisticVariable([o1, o2, o3, o4, o5], concept="Wartosc X",
                                                                 universe_of_discourse=[-1.0, 1.0]))

        j1 = FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=-0.1), term="Dol")
        j2 = FuzzySet(function=Trapezoidal_MF(a=0.1, b=0.3, c=1, d=1), term="Gora")
        FS.add_linguistic_variable("Y", LinguisticVariable([j1, j2], concept="Wartosc Y",
                                                                  universe_of_discourse=[-1.0, 1.0]))

        k1 = FuzzySet(function=Trapezoidal_MF(a=-8, b=-8, c=-4, d=-2), term="BNZgodnie")
        k2 = FuzzySet(function=Triangular_MF(a=-3, b=-1.5, c=0.5), term="TNZgodnie")
        k3 = FuzzySet(function=Triangular_MF(a=-0.5, b=0, c=0.5), term="Brak")
        k4 = FuzzySet(function=Triangular_MF(a=0.5, b=1.5, c=3), term="TZgodnie")
        k5 = FuzzySet(function=Trapezoidal_MF(a=2, b=4, c=8, d=8), term="BZgodnie")
        FS.add_linguistic_variable("AngV", LinguisticVariable([k1, k2, k3, k4, k5], concept="Angular v",
                                                           universe_of_discourse=[-8.0, 8.0]))

        # Zdefiniuj wartości i zmienne lingwistyczne dla wyjść systemu
        n1 = FuzzySet(function=Trapezoidal_MF(a=-2.0, b=-2, c=-1.5, d=-1), term="NZgodnie")
        n2 = FuzzySet(function=Trapezoidal_MF(a=1, b=1.5, c=2, d=2), term="Zgodnie")
        FS.add_linguistic_variable("Torque", LinguisticVariable([n1, n2],
                                                                 universe_of_discourse=[-2.0, 2.0]))

        # Zdefiniuj reguły systemu
        r1 = "IF (Y IS Dol) AND ((X IS TLewo) OR (X IS BLewo)) AND ((AngV IS TZgodnie) OR (AngV IS BZgodnie)) THEN (Torque IS Zgodnie)"
        r2 = "IF (Y IS Dol) AND ((X IS TPrawo)OR (X IS BPrawo)) AND ((AngV IS TNZgodnie) OR (AngV IS BNZgodnie)) THEN (Torque IS NZgodnie)"

        r3 = "IF (Y IS Gora) AND(X IS Srodek) AND ((AngV IS BZgodnie) OR (AngV IS TZgodnie)) THEN (Torque IS NZgodnie)"
        r4 = "IF (Y IS Gora) AND(X IS Srodek)  AND ((AngV IS BNZgodnie) OR (AngV IS TNZgodnie)) THEN (Torque IS Zgodnie)"
        r5 = "IF (Y IS Gora) AND (X IS TLewo) AND (AngV IS Brak) THEN (Torque IS NZgodnie)"
        r6 = "IF (Y IS Gora) AND (X IS TPrawo) AND (AngV IS Brak) THEN (Torque IS Zgodnie)"
        FS.add_rules([r1, r2, r3, r4, r5, r6])

        return FS

    def get_action(self, observation):
        for k, v in zip(self.observation_keys, observation):
            self.fis.set_variable(k, v)

        res = self.fis.Mamdani_inference([k for k in self.output_keys])
        return [res[k] for k in self.output_keys]
