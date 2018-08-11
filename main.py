import gym
import tensorflow as tf
import matplotlib.pyplot as plt

import beta_policy as beta


def main(args):

    env = gym.envs.make("MountainCarContinuous-v0")
    env.observation_space.sample()

    global_step = tf.Variable(0, name="global_step", trainable=False)

    sess = tf.Session()

    actor = beta.Actor(learning_rate=0.001)
    critic = beta.Critic(learning_rate=0.1)
    sess.run(tf.global_variables_initializer())

    episode_rewards = beta.run_model(sess, env, actor, critic, 100, discount_factor=0.995)
    plt.plot(list(range(len(episode_rewards))), episode_rewards)
    plt.show()


if __name__ == '__main__':
    tf.app.run()
