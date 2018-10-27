import torch 
import numpy as np


def unity_env_fitness_func(env, network, device, normalizer, train_mode=True):
    """
    Runs one episode for the given unity-environment and network.
    The collected episode reward works as the fitness indicator.
    If more than one reward is given, the max value is provided as the fitness value.
    """
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    states = env_info.vector_observations
    episode_reward = np.zeros(2)
    while True:
        states = np.reshape(states, (1, len(env_info.agents) * env_info.vector_observations.shape[1]))
        states = normalizer.normalize(states)
        states = torch.from_numpy(states).float().to(device)
        actions = network(states).cpu().detach().numpy()
        actions = np.reshape(actions, (len(env_info.agents), env.brains[brain_name].vector_action_space_size))
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        episode_reward += rewards
        states = next_states
        if np.any(dones):
            break

    return max(episode_reward)
