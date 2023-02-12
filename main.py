import gym
import time
from typing import Type
from pendulum_controllers import RandomController, TestController, FuzzyController, Controller


def episode(env, max_steps=100, controllercls: Type[Controller] = RandomController, debug_mode=False):
    total_reward = 0.
    controller = controllercls(action_space=env.action_space)

    observation = env.reset()
    for t in range(max_steps):
        env.render()
        action = controller.get_action(observation)
        if debug_mode:
            print(f"Step: {t}")
            print(f"\tObservation: {observation}")
            print(f"\tAction: {action}")
            time.sleep(0.3)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Episode finished after {t + 1} timesteps, with total reward = {total_reward}")
    return total_reward


if __name__ == '__main__':

    # env = gym.make('LunarLanderContinuous-v2') - na 4 punkty z domu
    env = gym.make('Pendulum-v1')       # praca w czasie labu, lub w domu na 3 punkty

    n_episodes = 5
    rewards = [episode(env, 200, debug_mode=False, controllercls=FuzzyController)
               for _ in range(n_episodes)]
    print(f"Avg total reward = {sum(rewards)/len(rewards)}")
    env.close()
