from petab_select.model import ModelStandard
from petab_select.models import ModelsStandard
from petab_select.problem import ProblemStandard

ModelStandard.save_schema("model.yaml")
ModelsStandard.save_schema("models.yaml")
ProblemStandard.save_schema("problem.yaml")
