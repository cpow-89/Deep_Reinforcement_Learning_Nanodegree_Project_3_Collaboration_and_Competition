import numpy as np


def evaluate(agent, env, num_test_runs=3):
    brain_name = env.brain_names[0]

    for episode in range(num_test_runs):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(2)
        while True:
            # actions = agent.act(state)
            actions = np.random.randn(2, 2)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            if np.any(dones):
                break

        print("Score at Episode {}: {}".format(episode, np.mean(scores)))
