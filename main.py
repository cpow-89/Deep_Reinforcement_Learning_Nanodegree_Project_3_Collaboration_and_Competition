import os
import json
from unityagents import UnityEnvironment
import session


def main():
    # load config files
    with open(os.path.join(".", "configs", "tennis_linux.json"), "r") as read_file:
        config = json.load(read_file)

    env = UnityEnvironment(file_name=os.path.join(*config["env_path"]))
    session.evaluate(None, env, num_test_runs=1)
    env.close()


if __name__ == '__main__':
    main()
