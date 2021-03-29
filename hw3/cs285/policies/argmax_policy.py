import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):   
        if len(obs.shape) <= 3:
            obs = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_a = self.critic.qa_values(obs)
        actions = q_a.argmax(1)

        return actions.squeeze()
