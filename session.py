from population import Population
import fitness
import torch


def train(env, normalizer, config):
    population = Population(config, normalizer)
    best_net = population.evolve(env)

    return best_net


def evaluate(net, env, normalizer, config, num_test_runs=3):
    fitness_func = getattr(fitness, config["fitness_func"])
    for episode in range(num_test_runs):
        scores = fitness_func(env, net, torch.device(config["device"]), normalizer, False)
        print("Score at Episode {}: {}".format(episode, scores))
