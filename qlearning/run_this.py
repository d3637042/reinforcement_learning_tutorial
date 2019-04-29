
import gym
from RL_brain import QLearningTable
import numpy as np
global state_max, state_min
def update():
    for episode in range(5000): #sort of converges at ~4000 episodes but can't be stable due to discretized observation
        # initial observation
        observation = env.reset()
        step_count = 0
        while True:

            # fresh env
            if episode % 10 == 0:
                env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(discretize_state(observation)))

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            reward = reward if not done or step_count == 199 else -100
            # RL learn from this transition
            RL.learn(str(discretize_state(observation)), action, reward, str(discretize_state(observation_)))

            # swap observation
            observation = observation_
            step_count += 1
            # break while loop when end of this episode
            if done:
                print("episode ", episode, "done after ", step_count, " steps")
                break

    # end of game
    while True:
        input("Press Enter to continue...")
        observation = env.reset()
        while True:

            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(discretize_state(observation)))

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)

            # swap observation
            observation = observation_
            if done:
                print("done after ", step_count, " steps")
                break

    

def get_range():
    state_dim = env.observation_space.shape[0]
    state_max = np.ones(state_dim)*-np.inf
    state_min = np.ones(state_dim)*np.inf
    for test_ep in range(10000):
        # initial observation
        observation = env.reset()
        
        while True:

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(env.action_space.sample())
            for i in range(state_dim):
                if observation_[i] > state_max[i]:
                    state_max[i] = observation_[i]
                if observation_[i] < state_min[i]:
                    state_min[i] = observation_[i]

            # break while loop when end of this episode
            if done:
                break
    print(state_max, state_min)
    return state_max, state_min



def discretize_state(state):
    discrete_num = 10
    state_dim = env.observation_space.shape[0]
    dis_state = np.ones(state_dim)
    for i in range(state_dim):
        state_range = env.observation_space.high[i] - env.observation_space.low[i]
        # check if range is inf 
        if state_range > 1000000:
            if state[i] > state_max[i]:
                dis_state[i] = int((state_max[i]-state_min[i])/((state_max[i]-state_min[i])/discrete_num))
            if state[i] < state_min[i]:
                dis_state[i] = int((state_max[i]-state_min[i])/((state_max[i]-state_min[i])/discrete_num))
            dis_state[i] = int((state[i]-state_min[i])/((state_max[i]-state_min[i])/discrete_num))
        else:
            dis_state[i] = int((state[i]-env.observation_space.low[i])/((env.observation_space.high[i]-env.observation_space.low[i])/discrete_num))
    return dis_state


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    RL = QLearningTable(actions=list(range(env.action_space.n)))
    state_max, state_min = get_range()
    update()
    