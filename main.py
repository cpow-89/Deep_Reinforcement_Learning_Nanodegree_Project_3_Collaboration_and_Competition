import os
import json
import session
import utils
from normalizer import OnlineNormalizer
from unityagents import UnityEnvironment
import network
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Extended Q-Learning with given config")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        metavar="",
                        required=True,
                        help="Config file name - file must be available as .json in ./configs")

    args = parser.parse_args()

    # load config files
    with open(os.path.join(".", "configs", args.config), "r") as read_file:
        config = json.load(read_file)

    env = UnityEnvironment(file_name=os.path.join(*config["env_path"]))
    normalizer = OnlineNormalizer(config["network"]["observation_size"])

    if config["run_training"]:
        elite_net = session.train(env, normalizer, config)
        checkpoint_dir = os.path.join(".", *config["checkpoint_dir"], config["env_name"])
        utils.save_state_dict(os.path.join(checkpoint_dir), elite_net.state_dict())
    else:
        trained_net = getattr(network, config["network"]["type"])(config["network"]).to(torch.device(config["device"]))
        checkpoint_dir = os.path.join(".", *config["checkpoint_dir"], config["env_name"])
        trained_net.load_state_dict(utils.load_latest_available_state_dict(os.path.join(checkpoint_dir, "*")))
        session.evaluate(trained_net, env, normalizer, config, num_test_runs=50)

    env.close()


if __name__ == '__main__':
    main()
