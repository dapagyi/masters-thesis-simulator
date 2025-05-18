from rl_playground.vrp.time_budgeting.custom_types import Action, Info, Observation

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
    return go_action if observation.remaining_route else wait_action
