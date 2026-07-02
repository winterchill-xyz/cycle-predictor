"""cycle_predictor — ML menstrual cycle predictor.

See PLAN.md for the roadmap and research/RESEARCH.md for the literature survey.
The canonical unit of data is one *cycle*: (user_id, cycle_number) with a
cycle_length_days label and optional covariates. Dataset adapters live in
`cycle_predictor.data`.
"""

__version__ = "0.0.1"
