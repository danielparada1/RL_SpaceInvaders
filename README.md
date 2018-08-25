# RL_SpaceInvaders
Training an agent using deep Q network and convolutional neural networks

trainAgent.ipynb : 
    imports all necessary files and depencies including the functions from our BrainDQN file
    preprocesses the images that are used as inputs
    loops forever to train the neural network
    creates a gif/video of a given number of frames
    
BrainDQN_Nature3L.py
    Contains all of the necessary functions to create the neural network and the training method for the network



Atari environment used by the game : (in particular the step function)
https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

The Atari Learning Interface, also used (from atary_env.py) by the game :
https://github.com/openai/atari-py/blob/master/atari_py/ale_python_interface.py
https://github.com/mgbellemare/Arcade-Learning-Environment

Paper discussing the theory behind Resnet :
https://arxiv.org/pdf/1512.03385.pdf

Sample code with simple instructions (which I was unable to make work) to run the code locally on the machine :
https://github.com/floodsung/DQN-Atari-Tensorflow

Keras source code documentation used for translating the original tensorflow based neural network into Keras :
https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L762

StackOverflow forum found while trying to find an explanation as to why the original sample code's spaceship went all the way to the right of the screen and stayed there (issue yet to be resolved):
https://ai.stackexchange.com/questions/2449/what-are-different-actions-in-action-space-of-environment-of-pong-v0-game-from

Atary-py Manual (looking for a way to reduce the action_space (currently Discrete(6)) to it's minimal requirement (Discrete(4)) ) :
https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf

# Researcher and Books
- [Csaba Szepesvari](https://sites.ualberta.ca/~szepesva/)
  + [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/RLBook.html)
