import time
from copy import deepcopy

import torch
import numpy as np

import utils


class MCTSNode:
    def __init__(self, parent, state, actions, forward_fn):
        self.name = "node" + str(time.time_ns())
        self.parent = parent
        self.state = state
        self.count = 0
        self.reward = 0
        self.is_terminal = state.is_terminal()
        self._forward_fn = forward_fn
        if self.is_terminal:
            self.actions = None
            self.children = None
        else:
            self.actions = actions
            self.children = [None] * len(actions)

    def run(self, iters, batch_size=1):
        i = 0
        while i < iters:
            v = self._tree_policy()
            reward = 0.0
            for _ in range(batch_size):
                reward += v._default_policy()
            reward /= batch_size
            v._backup(reward)
            i += 1
        return self.children_ucb1()

    def best_child_idx(self):
        idx = np.argmax(self.children_ucb1())
        return idx

    def children_ucb1(self):
        if not self.is_terminal:
            scores = []
            for child in self.children:
                if child is not None:
                    probs = []
                    bounds = []
                    # there may be stochastic outcomes for the same action
                    for outcome in child:
                        probs.append(outcome.count)
                        bounds.append(outcome.UCB1())
                    probs = np.array(probs)
                    bounds = np.array(bounds)
                    probs = probs/probs.sum()
                    scores.append((probs * bounds).sum())
                else:
                    scores.append(np.inf)
            return scores
        else:
            return None

    def children_yield(self):
        if not self.is_terminal:
            scores = []
            for child in self.children:
                if child is not None:
                    probs = []
                    yields = []
                    # there may be stochastic outcomes for the same action
                    for outcome in child:
                        probs.append(outcome.count)
                        yields.append(outcome.reward/outcome.count)
                    probs = np.array(probs)
                    yields = np.array(yields)
                    probs = probs / probs.sum()
                    scores.append((probs * yields).sum())
                else:
                    scores.append(0)
            scores = np.array(scores)
            return scores
        else:
            return None

    def UCB1(self):
        if self.parent is None:
            return None
        else:
            N = self.parent.count
            score = self.reward/self.count + np.sqrt((2*np.log(N)) / self.count)
            return score

    def plan(self):
        if self.is_terminal:
            return self.state, "Finish, p: 1.0 -> [" + ", ".join(self.state.stack) + "]", 1.0
        idx = self.best_child_idx()
        if self.children[idx] is None:
            print("Plan not found.")
            return self.state, self.actions[idx], 1.0
        elif len(self.children[idx]) == 1:
            child_state, child_plan, child_prob = self.children[idx][0].plan()
            return child_state, self.actions[idx]+", p: 1.0 -> [" + ", ".join(self.children[idx][0].state.stack) + "]\n"+child_plan, child_prob
        else:
            probs = []
            for out in self.children[idx]:
                probs.append(out.count)
            probs = np.array(probs)
            probs = probs / probs.sum()
            prob_max = np.argmax(probs)
            p = np.max(probs)
            child_state, child_plan, child_prob = self.children[idx][prob_max].plan()
            return child_state, self.actions[idx] + ", p: %.3f -> [" % p + ", ".join(self.children[idx][prob_max].state.stack) \
                + "]\n"+child_plan, p*child_prob

    def _expand(self):
        idx = self.children.index(None)
        action = self.actions[idx]
        # ACT HERE #
        next_state = self._forward_fn.forward(self.state, action)
        self.children[idx] = [MCTSNode(parent=self,
                                       state=next_state,
                                       actions=next_state.get_available_actions(),
                                       forward_fn=self._forward_fn)]
        ############
        return self.children[idx][0]

    def _tree_policy(self):
        # if there is an unexpanded node, first expand it.
        if self.is_terminal:
            return self
        if None in self.children:
            return self._expand()
        # else choose the best child by UCB1
        else:
            # have to change here
            idx = self.best_child_idx()
            next_state = self._forward_fn.forward(self.state, self.actions[idx])
            children_states = list(map(lambda x: x.state, self.children[idx]))
            result, out_idx = utils.in_array(next_state, children_states)
            if not result:
                self.children[idx].append(MCTSNode(parent=self,
                                                   state=next_state,
                                                   actions=next_state.get_available_actions(),
                                                   forward_fn=self._forward_fn))
                return self.children[idx][-1]._tree_policy()
            else:
                return self.children[idx][out_idx]._tree_policy()

    def _default_policy(self):
        if not self.is_terminal:
            random_action = np.random.choice(self.actions)
            # ACT HERE #
            next_state = self._forward_fn.forward(self.state, random_action)
            v = MCTSNode(parent=None, state=next_state, actions=next_state.get_available_actions(), forward_fn=self._forward_fn)
            ############
            return v._default_policy()
        else:
            return self.state.reward()

    def _backup(self, reward):
        if self.parent is not None:
            self.count += 1
            self.reward += reward
            self.parent._backup(reward)
        else:
            self.count += 1
            self.reward += reward

    def __repr__(self):
        state = str(self.state)
        children = []
        if not self.is_terminal:
            children_scores = list(map(lambda x: "%.2f" % x, self.children_ucb1()))
            for c in self.children:
                if c is None:
                    children.append("None")
                else:
                    outcomes = list(map(lambda x: x.name, c))
                    outcomes = "[" + ", ".join(outcomes) + "]"
                    children.append(outcomes)
        string = "Name: " + self.name + "\n"
        if self.parent:
            string += "Parent: " + self.parent.name + "\n"
        else:
            string += "Parent: None\n"
        if not self.is_terminal:
            string += "Children: [" + ", ".join(children) + "]\n"
            string += "Children UCB1: [" + ", ".join(children_scores) + "]\n"
        string += "State:\n" + state + "\n"
        string += "Reward: " + str(self.reward) + "\n"
        string += "Count: " + str(self.count) + "\n"
        string += "Terminal: " + str(self.is_terminal)
        return string


class State:
    def __init__(self, stack=[], inserts=[], picks=[], drops=[], goal=None):
        self.stack = stack
        self.inserts = inserts
        self.picks = picks
        self.drops = drops
        self.goal = goal

    def reward(self):
        height = (len(self.stack)-len(self.inserts))
        if self.goal[0] == height and self.goal[1] == len(self.stack):
            return 10
        else:
            return 0

    def get_available_actions(self):
        actions = []
        if len(self.stack) == 0:
            for obj in self.picks:
                actions.append("makebase-"+obj)
        elif len(self.picks) > 0:
            for obj in self.picks:
                actions.append("put-"+obj)
        return actions

    def is_terminal(self):
        if len(self.picks) == 0 or self.reward():
            return True
        else:
            return False

    def __repr__(self):
        string = "Stack: [" + ", ".join(self.stack) + "]\n"
        string += "Inserts: [" + ", ".join(self.inserts) + "]\n"
        string += "Drops: [" + ", ".join(self.drops) + "]\n"
        string += "Picks: [" + ", ".join(self.picks) + "]\n"
        return string

    def __hash__(self):
        return hash(self.__repr__())


class ForwardDynamics:
    def __init__(self, tree, codes1, codes2, stack_state, insert_state):
        self.tree = tree
        self.codes1 = codes1
        self.codes2 = codes2
        self.stack_idx = stack_state
        self.insert_idx = insert_state

    def forward(self, state, action):
        new_state = deepcopy(state)
        act, obj = action.split("-")
        if act == "makebase":
            new_state.picks.remove(obj)
            new_state.stack.append(obj)
        elif act == "put":
            idx1 = int(obj[-1]) - 1
            idx2 = int(state.stack[-1][-1]) - 1
            x = torch.cat([self.codes1[idx2], self.codes1[idx1], self.codes2[idx2, idx1].unsqueeze(0)], dim=-1).numpy()
            eff_prob = self.tree.predict_proba([x])[0]
            effect = np.random.multinomial(1, eff_prob)
            effect = np.argmax(effect)
            new_state.picks.remove(obj)
            # stack effect
            if effect == self.stack_idx:
                new_state.stack.append(obj)
            elif effect == self.insert_idx:
                new_state.stack.append(obj)
                new_state.inserts.append(obj)
            else:
                new_state.drops.append(obj)
        return new_state
