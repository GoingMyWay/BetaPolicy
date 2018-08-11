import itertools
import collections

import numpy as np
import tensorflow as tf
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.kernel_approximation import RBFSampler


def rbf_featurizer(env, sample_len=1000):
    examples = np.array([env.observation_space.sample() for _ in range(sample_len)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(examples)

    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    ])
    featurizer.fit(scaler.transform(examples))
    return featurizer, scaler


def get_feature(featurizer, scaler, state):
    scaled = scaler.transform([state])
    output_state = featurizer.transform(scaled)
    return output_state[0]


class Actor:
    def __init__(self, learning_rate=0.01, scope="actor"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.alpha = tf.layers.dense(inputs=tf.expand_dims(self.state, 0),
                                         units=1,
                                         kernel_initializer=tf.zeros_initializer)
            self.alpha = tf.nn.softplus(tf.squeeze(self.alpha)) + 1.

            self.beta = tf.layers.dense(inputs=tf.expand_dims(self.state, 0),
                                        units=1,
                                        kernel_initializer=tf.zeros_initializer)
            self.beta = tf.nn.softplus(tf.squeeze(self.beta)) + 1.

            self.beta_dist = tf.distributions.Beta(self.alpha, self.beta)
            action = self.beta_dist._sample_n(1)

            cliped_value = tf.clip_by_value(action, 0, 1)
            self.action = 2 * cliped_value - 1
            train_action = (cliped_value+1)/2.

            self.loss = -self.beta_dist.log_prob(train_action) * self.target
            self.loss -= 1e-2 * self.beta_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class Critic:
    def __init__(self, learning_rate=0.1, scope="critic"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.output_layer = tf.layers.dense(inputs=tf.expand_dims(self.state, 0),
                                                units=1,
                                                kernel_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def run_model(sess, env, actor, critic, num_episodes, discount_factor=1.0, play=False):

    featurizer, scaler = rbf_featurizer(env, sample_len=1000)

    Transition = collections.namedtuple("Transition",
                                        ["state", "action", "reward", "next_state", "done"])

    episode_rewards = [0] * num_episodes

    for i_episode in range(num_episodes):
        state = env.reset()

        episode = []
        reward_counter = 0

        for t in itertools.count():
            env.render()

            action = actor.predict(get_feature(featurizer, scaler, state), sess)
            next_state, reward, done, _ = env.step(action)

            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            reward_counter += reward

            if not play:
                value_next = critic.predict(get_feature(featurizer, scaler, next_state), sess)
                td_target = reward + discount_factor * value_next
                td_error = td_target - critic.predict(get_feature(featurizer, scaler, state), sess)

                critic.update(get_feature(featurizer, scaler, state), td_target, sess)
                actor.update(get_feature(featurizer, scaler, state), td_error, sess)

            print("\rStep {}, Episode {}/{} ({}), {}, {}".format(
                t, i_episode + 1, num_episodes,
                episode_rewards[i_episode - 1], reward, action[0]), end="")

            if done:
                break

            state = next_state

    return episode_rewards
