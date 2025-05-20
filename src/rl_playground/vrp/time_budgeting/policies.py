from itertools import combinations

from rl_playground.vrp.time_budgeting.custom_types import Action, Info, Observation
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv

go_action = Action(
    accepted_customers=[],
    wait_at_current_location=False,
)

wait_action = Action(
    accepted_customers=[],
    wait_at_current_location=True,
)


def reject_policy(observation: Observation, info: Info) -> Action:
    """Reject all new customers and go to the next customer in the route if there is one."""
    return go_action if info.remaining_route else wait_action


def greedy_policy(observation: Observation, info: Info, env: TimeBudgetingEnv) -> Action:
    if not info.new_customers:
        return go_action if info.remaining_route else wait_action

    for k in reversed(range(len(info.new_customers) + 1)):
        for accepted_customers in combinations(info.new_customers, k):
            action = Action(
                accepted_customers=list(accepted_customers),
                wait_at_current_location=False,
            )
            try:
                env.calculate_post_decison_state(action)
                return action
            except ValueError:
                continue

    raise ValueError("No valid action found.")
