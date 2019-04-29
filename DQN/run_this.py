import gym
from RL_brain import DeepQNetwork
#from DQN_modified import DeepQNetwork


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        step = 0
        while True:
            # fresh env

            if episode % 1 == 0:
                env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)
            reward = reward if not done or step == 199 else -100
            RL.store_transition(observation, action, reward, observation_)

            RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                RL.update_target()
                break
            step += 1
        print('episode:', episode, 'step: ', step)
    # end of game
    print('game over')
    while True:
        observation = env.reset()
        while True:
            # fresh env

            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)

            # break while loop when end of this episode
            if done:
                break
            step += 1

    #env.destroy()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    RL = DeepQNetwork(env.action_space.n, env.env.observation_space.shape[0],
                    learning_rate=0.001,
                    reward_decay=0.99,
                    e_greedy=0.001,
                    replace_target_iter=1000,
                    memory_size=2000,
                    batch_size=64,
                    e_greedy_decay=0.999, #1 if you want to do all e-greedy
                    output_graph=True,
                    )
    update()
    RL.plot_cost()