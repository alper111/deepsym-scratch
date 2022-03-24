import time
import uuid
from copy import deepcopy

import torch
import numpy as np

import utils


class MCTSNode:
    def __init__(self, parent, state, forward_fn):
        self.name = uuid.uuid4().hex
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
            self.actions = state.get_available_actions()
            self.children = [None] * len(self.actions)

    def run(self, iter_limit, time_limit, default_depth_limit=10, default_batch_size=1):
        i = 0
        start = time.time()
        end = time.time()
        time_elapsed = end - start
        start_node_count, _ = self._tree_stats()
        while (i < iter_limit) and (time_elapsed < time_limit):
            v = self._tree_policy()
            reward = 0.0

            # sequential
            for _ in range(default_batch_size):
                reward += v._default_policy(default_depth_limit)
            reward /= default_batch_size

            v._backup(reward)

            i += 1
            end = time.time()
            time_elapsed = end - start
            if i % 100 == 0:
                node_count, depth = self._tree_stats()
                print(f"Tree depth={depth}, node count={node_count}, node/sec={(node_count-start_node_count)/time_elapsed:.2f}, best reward={self.reward/self.count}")

        return self.children_yield()

    def best_child_idx(self):
        idx = np.argmax(self.children_ucb1())
        return idx

    def best_child_for_plan(self):
        idx = np.argmax(self.children_yield())
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
            return self.state, "", [], 1.0
        idx = self.best_child_for_plan()
        if self.children[idx] is None:
            print("Plan not found.")
            return self.state, "", [(idx, 0)], 1.0
        elif len(self.children[idx]) == 1:
            child_state, child_plan_txt, child_plan, child_prob = self.children[idx][0].plan()
            return child_state, "-".join(filter(None, [self.actions[idx], child_plan_txt])), [(idx, 0)]+child_plan, child_prob
        else:
            probs = []
            for out in self.children[idx]:
                probs.append(out.count)
            probs = np.array(probs)
            probs = probs / probs.sum()
            prob_max = np.argmax(probs)
            p = np.max(probs)
            child_state, child_plan_txt, child_plan, child_prob = self.children[idx][prob_max].plan()
            return child_state, "-".join(filter(None, [self.actions[idx], child_plan_txt])), [(idx, prob_max)]+child_plan, p*child_prob

    def _expand(self):
        idx = self.children.index(None)
        action = self.actions[idx]
        # ACT HERE #
        next_state = self._forward_fn.forward(self.state, action)
        self.children[idx] = [MCTSNode(parent=self,
                                       state=next_state,
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
                                                   forward_fn=self._forward_fn))
                return self.children[idx][-1]._tree_policy()
            else:
                return self.children[idx][out_idx]._tree_policy()

    def _default_policy(self, depth_limit):
        if (not self.is_terminal) and (depth_limit > 0):
            random_action = np.random.choice(self.actions)
            # ACT HERE #
            next_state = self._forward_fn.forward(self.state, random_action)
            v = MCTSNode(parent=None, state=next_state, forward_fn=self._forward_fn)
            ############
            return v._default_policy(depth_limit-1)
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

    def _tree_stats(self):
        if self.is_terminal:
            return 1, 0

        children_depths = []
        total_nodes = 1
        for c in self.children:
            if c is None:
                children_depths.append(0)
            elif len(c) == 1:
                nodes, depth = c[0]._tree_stats()
                children_depths.append(depth)
                total_nodes += nodes
            else:
                gchild_depths = []
                for c_i in c:
                    nodes, depth = c_i._tree_stats()
                    gchild_depths.append(depth)
                    total_nodes += nodes
                children_depths.append(max(gchild_depths)+1)
        return total_nodes, max(children_depths)+1


class MCTSState:
    def __init__(self, goal):
        pass

    def reward(self):
        pass

    def get_available_actions(self):
        pass

    def is_terminal(self):
        pass

    def __repr__(self):
        pass

    def is_equal(self, other):
        pass


class MCTSForward:
    def __init__(self):
        pass

    def forward(self, state, action):
        pass


class MNISTState(MCTSState):
    def __init__(self, precond, goal):
        self.precond = precond
        self.goal = goal
        self.cossim = torch.nn.CosineSimilarity(dim=-1)

    def reward(self):
        if self.is_terminal():
            return 1000
        else:
            return 10*self.cossim(self.precond.flatten(), self.goal.flatten())

    def get_available_actions(self):
        return ["move_right", "move_up", "move_left", "move_down"]

    def is_terminal(self):
        if self.cossim(self.precond.flatten(), self.goal.flatten()) > 0.95:
            return True
        else:
            return False

    def __repr__(self):
        return self.precond.__repr__()

    def is_equal(self, other):
        if self.cossim(self.precond.flatten(), other.precond.flatten()) > 0.95:
            return True
        else:
            return False


class MNISTForward(MCTSForward):
    def __init__(self, deepsym_module):
        self.model = deepsym_module
        self.action_dict = {
            "move_right": torch.tensor([[1., 0., 0., 0.]], device=self.model.device),
            "move_up": torch.tensor([[0., 1., 0., 0.]], device=self.model.device),
            "move_left": torch.tensor([[0., 0., 1., 0.]], device=self.model.device),
            "move_down": torch.tensor([[0., 0., 0., 1.]], device=self.model.device)
        }

    def forward(self, state, action):
        state_dict = {
            "state": state.precond,
            "action": self.action_dict[action]
        }
        _, effect = self.model.forward(state_dict, eval_mode=True)

        new_precond = (state.precond + effect).clamp(0., 1.)
        new_state = MNISTState(new_precond, state.goal)
        return new_state


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
            # x = torch.cat([self.codes1[idx2], self.codes1[idx1], self.codes2[idx2, idx1].unsqueeze(0)], dim=-1).numpy()
            x = torch.cat([self.codes2[idx2, idx1].unsqueeze(0), self.codes1[idx2], self.codes1[idx1], torch.tensor([1.0])], dim=-1).numpy()
            eff_prob = self.tree.predict_proba([x])[0]
            effect = np.random.multinomial(1, eff_prob)
            effect = np.argmax(effect)
            new_state.picks.remove(obj)
            # stack effect
            if effect in self.stack_idx:
                new_state.stack.append(obj)
            elif effect in self.insert_idx:
                new_state.stack.append(obj)
                new_state.inserts.append(obj)
            else:
                new_state.drops.append(obj)
        return new_state
