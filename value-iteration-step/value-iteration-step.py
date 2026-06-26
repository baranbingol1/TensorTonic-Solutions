def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    new_values = values.copy()
    num_states = len(values)

    for state in range(num_states):
        new_value = max(
            rewards[state][action] + gamma * sum(
                transitions[state][action][next_state] * values[next_state]
                for next_state in range(num_states)
            )
            for action in range(len(transitions[state]))
        )

        new_values[state] = new_value

    return new_values