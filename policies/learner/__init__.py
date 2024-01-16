from .Learner import Learner
from .AtariLearner import AtariLearner
from .GeneralizeLearner import GeneralizeLearner
from .MetaLearner import MetaLearner
from .RmdpLearner import RmdpLearner

LEARNER_CLASS = {
    "meta": MetaLearner,
    "pomdp": Learner,
    "credit": Learner,
    "rmdp": RmdpLearner,
    "generalize": GeneralizeLearner,
    "atari": AtariLearner,
}
