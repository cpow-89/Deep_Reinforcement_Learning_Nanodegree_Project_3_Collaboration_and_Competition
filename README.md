# Deep Reinforcement Learning Nanodegree: Project 3 - Collaboration and Competition

This project includes the code for a simplified version of the deep genetic algorithm introduced in the paper "Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning" by Uber AI Labs. I wrote it to solve the Project 3 - Collaboration and Competition of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) @ Udacity. 
 
For more information on the implemented features refer to "Report.ipynb". The notebook includes a summary of all essential concepts used in the code.


### Project 3 - Collaboration and Competition - Details:

In this project, two agents should be trained to control rackets to bounce a ball over a net. To maximize the reward, the agents need to learn how to hit the ball over the net and also how to avoid to let the ball hit the ground or fly out of bounds.

[//]: # (Image References)

#### Random Agent

[image1]: https://raw.githubusercontent.com/cpow-89/Deep_Reinforcement_Learning_Nanodegree_Project_3_Collaboration_and_Competition/master/images/untrained_agent.gif?token=AmwnwjlLSB-Io9VjZcw4NbldZTsT0_pDks5b4ecxwA%3D%3D "Random Agent"

![Random Agent][image1]


#### Trained Agent

[image2]: https://raw.githubusercontent.com/cpow-89/Deep_Reinforcement_Learning_Nanodegree_Project_3_Collaboration_and_Competition/master/images/trained_agent.gif?token=AmwnwrvTIWlWKXU-Zj2nPZNWK6bxFFw2ks5b4eb3wA%3D%3D "Trained Agent"
![Trained Agent][image2]

##### Reward:
- a reward of +0.1 is provided if an agent hits the ball over the net
- a reward of -0.01 is provided if an agent lets a ball hit the ground or if an agent hits the ball out of bounds

##### Search Space
- the observation space has 48 dimensions 
     - Two agents times 24 inputs
- the action space has four dimensions
    - Two agents times 2 actions
    - every action is a continuous number between -1 and 1

##### Task
- the task is episodic
- two agents control rackets to bounce a ball over a net
- agents need to learn how to hit the ball over the net 
- agents need to learn how to avoid letting the ball hit the ground or fly out of bounds
- to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes
        

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Download the environment from one of the links below and place it into \p3_collab-compet\

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    
- your folder should now look something like this:

p3_collab-compet\Tennis_Linux\ <br>
&nbsp;&nbsp;&nbsp;&nbsp; \Tennis_Data  <br>
&nbsp;&nbsp;&nbsp;&nbsp; \Tennis.x86<br>
&nbsp;&nbsp;&nbsp;&nbsp; \Tennis.x86_64<br>

3. Install Sourcecode dependencies

> conda install -c pytorch pytorch <br>
> conda install -c anaconda numpy <br>
> pip install tensorboardX

- unityagents is also required
    - an easy way to get this is to install the Deep Reinforcement Learning Nanodegree with its dependencies
    
> git clone https://github.com/udacity/deep-reinforcement-learning.git<br>
> cd deep-reinforcement-learning/python<br>
> pip install .<br>

### How to run the project

You can run the project by running the main.py file through the console.
- open the console and run: python main.py -c "your_config_file.json" 
- to train the agent from scratch set "run_training" in the config file to true
- to run the pre-trained agent set "run_training" in the config file to false

optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "tennis_linux.json" 
