"""
PPO(Proximal Policy Optimization) Algorithm is implementd in two steps:

1. Training data genreation
2. Training/ Update weights of Neural network
"""
import torch
import random

class PPO:

    def __init__(self, config):

        self.rl_env = config["rl_env"]
        self.policy_network = config["policy_network"]
        self.value_network = config["value_network"]
        self.max_trajectory_length = config["max_trajectory_length"]
        self.discount_factor = config["discount_factor"]
        self.iterations = config["iterations"]
        self.epochs = config["epochs"]
        self.mini_batch_size = config["mini_batch_size"]
        self.eta = config["eta"]
        self.optimizer = config["optimizer"]
        self.coeff1 = config["coeff1"]
        self.coeff2 = config["coeff2"]
        
        self.train_data = []

    def generate_samples(self):
        for _ in range(self.iterations):
            # reset the env and get the first state
            s_t = self.rl_env.reset()
            recieved_rewards = []
            trajectory_sample = []
            for _ in range(self.max_trajectory_length):
                policy_probs = self.policy_network(s_t)
                a_t = random.sample(policy_probs, k=1)

                rt, s_t_1 = self.rl_env.step(a_t)
                recieved_rewards.append(rt)

                v_t = self.value_network(s_t)

                trajectory_sample.append({
                    "s_t": s_t,
                    "old_policy_prob" : policy_probs.index(a_t),
                    "a_t": a_t,
                    "rt" : rt,
                    "v_t": v_t
                    })

                s_t = s_t_1
            
            # calculate V_target and Advantage for each state
            target_reward = None
            
            for i in range(len(recieved_rewards),0,-1):
                true_reward = recieved_rewards[i]
                next_value = target_reward if target_reward else 0
                discounted_reward_for_current_step = true_reward + self.discount_factor * next_value
                target_reward = discounted_reward_for_current_step

                curr_sample = trajectory_sample[i]
                advantage_for_current_step = discounted_reward_for_current_step - curr_sample["v_t"] 

                curr_sample["v_target"] = discounted_reward_for_current_step
                curr_sample["advantage_t"] = advantage_for_current_step
            
            self.train_data.extend(trajectory_sample)
    
    def train(self):
        self.optimizer.zero_gradient()

        for _ in range(self.epochs):

            train_data = random.shuffle(self.train_data)
            total_losses = 0
            for i in range(self.mini_batch_size):
                data = random.sample(train_data, k=1)
                policy_probs = self.policy_network(data["s_t"])
                new_policy_prob = policy_probs.index(data["a_t"])

                v_t = self.value_network(data["s_t"])

                # calculate clip loss
                prob_ratio = new_policy_prob/ data["old_policy_prob"]

                if prob_ratio < 1-self.eta:
                    clipped_prob = 1-self.eta
                elif prob_ratio > 1 + self.eta:
                    clipped_prob = 1 + self.eta
                else:
                    clipped_prob = prob_ratio

                loss_clip = min( prob_ratio* data["advantage_t"], clipped_prob* data["advantage_t"])
                loss_value = (data["v_target"] - v_t)**2
                loss_entropy = -1*new_policy_prob*torch.log(new_policy_prob)

                loss = -1* (loss_clip + self.coeff1 * loss_value + self.coeff2 * loss_entropy)
                total_losses += loss
            
            self.optimizer.backward(total_losses/self.mini_batch_size)
            self.optimizer.update()