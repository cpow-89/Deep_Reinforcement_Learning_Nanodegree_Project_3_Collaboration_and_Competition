import network
import numpy as np
import mutation
import fitness
from collections import deque
import os
import torch
from tensorboardX import SummaryWriter
import utils


class Population:
    def __init__(self, config, normalizer):
        self.config = config
        self.device = torch.device(config["device"])
        self.normalizer = normalizer
        self.fitness_func = getattr(fitness, self.config["fitness_func"])
        self.population = None
        self.elite = None

        self.writer = SummaryWriter(os.path.join(".", *self.config["monitor_dir"],
                                                 self.config["env_name"], utils.get_current_date_time()))
        self.elite_scores = deque(maxlen=100)
        self.best_mean = None
        self.gen_idx = 1

    def _create_init_population(self, env):
        networks = [getattr(network, self.config["network"]["type"])(self.config["network"]).to(self.device)
                    for _ in range(self.config["population_size"])]
        return [(net, self.fitness_func(env, net, self.device, self.normalizer)) for net in networks]

    def _write_statistics(self):

        self.elite_scores.append(self.population[0][1])

        print("Generation: {} - Elite scores mean over 100 episodes: {}".format(self.gen_idx,
                                                                                np.mean(self.elite_scores)))
        self.writer.add_scalar("Elite_scores_mean_over_100_episodes", np.mean(self.elite_scores), self.gen_idx)

    def _env_solved(self):
        return np.mean(self.elite_scores) >= self.config["mean_score_to_solve"] and len(self.elite_scores) >= 100

    def _create_next_generation(self, env):
        prev_population = self.population
        self._elitism(env)
        self.population = [self.elite]
        for _ in range(self.config["population_size"] - 1):
            parent_idx = np.random.randint(0, self.config["parent_count"])
            parent = prev_population[parent_idx][0]
            child = mutation.gaussian_noise_mutation(parent, self.config["noise_std"], self.device)
            fitness_val = self.fitness_func(env, child, self.device, self.normalizer)
            self.population.append((child, fitness_val))

    def _elitism(self, env):
        elite_pool = []
        for parent in self.population:
            mean_reward = np.mean([self.fitness_func(env, parent[0], self.device, self.normalizer)
                                   for _ in range(self.config["elite_evaluation_count"])])
            elite_pool.append((parent[0], mean_reward))

        elite_pool = self._sort_population_by_fitness(elite_pool)
        self.elite = (elite_pool[0][0], elite_pool[0][1])

    @staticmethod
    def _sort_population_by_fitness(population):
        population.sort(key=lambda p: p[1], reverse=True)
        return population

    def evolve(self, env):
        self.population = self._create_init_population(env)
        while True:
            self.population = self._sort_population_by_fitness(self.population)

            self._write_statistics()

            if self.best_mean is None or np.mean(self.elite_scores) > self.best_mean:
                self.best_mean = np.mean(self.elite_scores)
                checkpoint_dir = os.path.join(".", *self.config["checkpoint_dir"], self.config["env_name"])
                utils.save_state_dict(os.path.join(checkpoint_dir), self.population[0][0].state_dict())

            if self._env_solved():
                print("Environment solved in {}".format(self.gen_idx))
                break

            self._create_next_generation(env)
            self.gen_idx += 1

        return self.elite[0]
