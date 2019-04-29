from ppo import PPO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

all_ep_r = []
EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.99
BATCH = 16

if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	ppo = PPO(env.action_space.n, env.observation_space.shape[0])
	for ep in range(EP_MAX):
		s = env.reset()
		buffer_s, buffer_a, buffer_r = [], [], []
		ep_r = 0
		for t in range(EP_LEN):    # in one episode
			env.render()
			a = ppo.choose_action(s)
			s_, r, done, _ = env.step(a)
			#print(s_, r,)
			r = r if not done or t == 199 else -100
			buffer_s.append(s)
			buffer_a.append(a)
			buffer_r.append(r)    # normalize reward, find to be useful
			s = s_
			ep_r += r

			# update ppo
			if (t+1) % BATCH == 0 or t == EP_LEN-1 or done:
				v_s_ = ppo.get_v(s_)
				discounted_r = []
				for r in buffer_r[::-1]:
					v_s_ = r + GAMMA * v_s_
					discounted_r.append(v_s_)
				discounted_r.reverse()

				bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
				buffer_s, buffer_a, buffer_r = [], [], []
				ppo.update(bs, ba, br)
			if done:
				ppo.update_target()
				break
		if ep == 0: all_ep_r.append(ep_r)
		else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
		print(
			'Ep: %i' % ep,
			'|step: %i' % t,
			'|Ep_r: %i' % ep_r)

	plt.plot(np.arange(len(all_ep_r)), all_ep_r)
	plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()