import torch
import numpy as np
from sklearn.tree import _tree


def kmeans(x, k, centroids=None, max_iter=None, epsilon=0.01):
    """
    x: data set of size (n, d) where n is the sample size.
    k: number of clusters
    centroids (optional): initial centroids
    max_iter (optional): maximum number of iterations
    epsilon (optional): error tolerance
    returns
    centroids: centroids found by k-means algorithm
    next_assigns: assignment vector
    mse: mean squared error
    """
    if centroids is None:
        centroids = torch.zeros(k, x.shape[1], device=x.device)
        prev_assigns = torch.randint(0, k, (x.shape[0],), device=x.device)
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)

    distances = torch.cdist(centroids, x) ** 2
    prev_assigns = torch.argmin(distances, dim=0)

    it = 0
    prev_mse = distances[prev_assigns, torch.arange(x.shape[0])].mean()
    while True:
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)
        distances = torch.cdist(centroids, x) ** 2
        next_assigns = torch.argmin(distances, dim=0)
        if (next_assigns == prev_assigns).all():
            break
        else:
            prev_assigns = next_assigns
        it += 1
        mse = distances[next_assigns, torch.arange(x.shape[0])].mean()
        error = abs(prev_mse-mse)/prev_mse
        prev_mse = mse
        print("iteration: %d, mse: %.3f" % (it, prev_mse.item()))

        if it == max_iter:
            break
        if error < epsilon:
            break

    return centroids, next_assigns, prev_mse, it


def tree_to_code(tree, effect_names, obj_names, probabilistic):
    tree_ = tree.tree_

    def recurse(node, rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            left = rules.copy()
            right = rules.copy()
            left.append(-(tree_.feature[node]+1))
            right.append(tree_.feature[node]+1)
            rules_from_left = recurse(tree_.children_left[node], left)
            rules_from_right = recurse(tree_.children_right[node], right)
            rules = np.concatenate([rules_from_left, rules_from_right])
            return rules
        else:
            print("rules:", rules)
            obj1_list, obj2_list, comparison = rule_to_code(rules, obj_names)
            precond = ":precondition (and (not (stacked)) (not (inserted)) (pickloc ?above) (stackloc ?below) "
            if len(obj1_list) > 1:
                precond += "(or"
                for obj1 in obj1_list:
                    precond += " (%s ?below)" % obj1
                precond += ") "
            else:
                precond += "(%s ?below) " % obj1_list[0]

            if len(obj2_list) > 1:
                precond += "(or"
                for obj2 in obj2_list:
                    precond += " (%s ?above)" % obj2
                precond += ")"
            else:
                precond += "(%s ?above)" % obj2_list[0]

            if comparison is not None:
                precond += " %s" % comparison
            precond += ")"

            print(precond)
            counts = tree_.value[node][0]
            effect = ":effect (and "
            if probabilistic:
                effect += "(probabilistic"
                # this shenanigan is needed because probabilities add up to more than one.
                probs = (counts / counts.sum())
                probs = (probs * 100000).round().astype(np.int)
                ptotal = probs.sum()
                if ptotal > 100000:
                    residual = ptotal - 100000
                    probs[np.argmax(probs)] -= residual

                for i in range(len(counts)):
                    if probs[i] == 0:
                        continue
                    elif probs[i] != 100000:
                        effect += "\n\t\t\t\t 0.%05d " % (probs[i])
                    else:
                        effect += "\n\t\t\t\t 1.00000 "

                    if effect_names[i][0] == "s":
                        effect += "(and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))"
                    elif effect_names[i][0] == "i":
                        effect += "(and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))"
                    else:
                        effect += "(%s)" % (effect_names[i])
                effect += ")"
            else:
                idx = counts.argmax()
                if effect_names[idx][0] == "s":
                    effect += "\n\t\t\t\t (and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))"
                elif effect_names[idx][0] == "i":
                    effect += "(and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))"
                else:
                    effect += "\n\t\t\t\t (%s)" % (effect_names[idx])
            effect += "\n\t\t\t\t(not (pickloc ?above)))"
            return np.array([[precond, effect]])
    return recurse(0, [])


def rule_to_code(rule, obj_names):
    absrules = np.abs(rule).tolist()
    indices = []
    for x in range(1, 6):
        if x in absrules:
            indices.append(absrules.index(x))
        else:
            indices.append(-1)

    possible_obj_1 = list(obj_names.keys())
    possible_obj_2 = list(obj_names.keys())
    for i, idx in enumerate(indices[1:3]):
        if idx == -1:
            continue
        # sign = np.sign(rule[idx])
        sign = 0 if np.sign(rule[idx]) == -1 else 1
        # burada artik obje kategorileri 0,1 uzerinden
        # fix this
        possible_obj_1 = list(filter(lambda x: x[i] == sign, possible_obj_1))

    for i, idx in enumerate(indices[3:]):
        if idx == -1:
            continue
        # sign = np.sign(rule[idx])
        sign = 0 if np.sign(rule[idx]) == -1 else 1
        possible_obj_2 = list(filter(lambda x: x[i] == sign, possible_obj_2))

    obj1_list = [obj_names[x] for x in possible_obj_1]
    obj2_list = [obj_names[x] for x in possible_obj_2]

    if indices[0] == -1:
        # comparison = "(or (relation0 ?below ?above) (relation1 ?below ?above))"
        comparison = ""
    else:
        sign = np.sign(rule[indices[0]])
        if sign == -1:
            comparison = "(relation0 ?below ?above)"
        elif sign == 1:
            comparison = "(relation1 ?below ?above)"
        else:
            print("hata")
            exit()

    return obj1_list, obj2_list, comparison


def tree_to_code_v2(tree, actions, probabilistic, num_bits, threshold=0):
    tree_ = tree.tree_

    def recurse(node, branch):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            left = branch.copy()
            right = branch.copy()
            left.append(-(tree_.feature[node]+1))
            right.append(tree_.feature[node]+1)
            rules_from_left = recurse(tree_.children_left[node], left)
            rules_from_right = recurse(tree_.children_right[node], right)
            rules = np.concatenate([rules_from_left, rules_from_right])
            return rules
        else:
            counts = tree_.value[node][0]
            if counts.sum() < threshold:
                return np.array([("", "", "")])
            precond = ":precondition (and "
            action_rules = []
            for f_i in branch:
                if abs(f_i) in actions.keys():
                    action_rules.append(f_i)
                    continue

                if f_i < 0:
                    z_i = "(not (z%d)) " % (abs(f_i)-1)
                else:
                    z_i = "(z%d) " % (abs(f_i)-1)
                precond += z_i

            precond = precond[:-1] + ")"

            effect = ":effect "
            if probabilistic:
                effect += "(probabilistic"
                # this shenanigan is needed because probabilities add up to more than one.
                probs = (counts / counts.sum())
                probs = (probs * 100000).round().astype(np.int)
                ptotal = probs.sum()
                if ptotal > 100000:
                    residual = ptotal - 100000
                    probs[np.argmax(probs)] -= residual

                for i in range(len(counts)):
                    if probs[i] == 0:
                        continue
                    if probs[i] != 100000:
                        effect += "\n\t\t\t\t 0.%05d " % (probs[i])
                    else:
                        effect += "\n\t\t\t\t 1.00000 "

                    e_i = decimal_to_binary(tree.classes_[i], length=num_bits)
                    effect += "(and "
                    for idx, e_ij in enumerate(e_i):
                        if e_ij == 0:
                            effect += "(not (z%d)) " % idx
                        else:
                            effect += "(z%d) " % idx
                    # do not include action features
                    # for a_i in action_features:
                    #     effect += "(not (z%d)) " % a_i
                    effect = effect[:-1] + ")"
                effect += ")"
            else:
                idx = counts.argmax()
                e_i = decimal_to_binary(tree.classes_[idx], length=num_bits)
                effect += "(and "
                for idx, e_ij in enumerate(e_i):
                    if e_ij == 0:
                        effect += "(not (z%d)) " % idx
                    else:
                        effect += "(z%d) " % idx
                # do not include action features
                # for a_i in action_features:
                #     effect += "(not (z%d)) " % a_i
                effect = effect[:-1] + ")"

            action_rules = np.array(action_rules)
            mask = action_rules > 0
            if mask.any():
                key = action_rules[mask][0]
                action_name = actions[key]
            else:
                action_name = ""
                for i in range(num_bits+1, num_bits+5):
                    if i not in abs(action_rules):
                        if len(action_name) == 0:
                            action_name = actions[i]
                        else:
                            action_name = action_name + "_or_" + actions[i]
            print(branch)
            print(action_rules, action_name)
            print(precond)

            return np.array([(precond, effect, action_name)])
    return recurse(0, [])


def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        total_num += p.shape.numel()
    return total_num


def decimal_to_binary(number, length=None):
    arr = []
    while number > 1:
        arr.append(number % 2)
        number = number // 2
    arr.append(number)
    arr = list(map(lambda x: 1 if x == 1 else 0, arr))
    if length is not None and len(arr) < length:
        pad = length - len(arr)
        for _ in range(pad):
            arr.append(0)
    return tuple(reversed(arr))


def decimal_to_binary_v2(number_list, length=None):
    binaries = [format(x, "0"+str(length)+"b") for x in number_list]
    return binaries


def binary_to_decimal(number):
    dec_number = 0
    for i, digit in enumerate(reversed(number)):
        multiplier = 2**i
        if int(digit) == 1:
            dec_number += multiplier
    return dec_number


def binary_to_decimal_tensor(x):
    N, D = x.shape
    dec_tensor = torch.zeros(x.shape[0], dtype=torch.int32)
    for i in reversed(range(D)):
        multiplier = 2**i
        dec_tensor += multiplier * x[:, D-i-1].int()
    return dec_tensor


# def in_array(element, array):
#     element_hash = hash(element)
#     for idx in range(len(array)):
#         if element_hash == hash(array[idx]):
#             return True, idx
#     return False, None


def in_array(element, array):
    for i, e_i in enumerate(array):
        if element.is_equal(e_i):
            return True, i
    return False, None


def return_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def cc_pix_avg(img, x, y):
    height, width = img.shape
    img[x, y] = False
    painted = [[x, y]]
    if x+1 < height and img[x+1, y]:
        painted += cc_pix_avg(img, x+1, y)
    if x-1 > 0 and img[x-1, y]:
        painted += cc_pix_avg(img, x-1, y)
    if y+1 < width and img[x, y+1]:
        painted += cc_pix_avg(img, x, y+1)
    if y-1 < 0 and img[x, y-1]:
        painted += cc_pix_avg(img, x, y-1)
    return painted


def find_objects(img, window_size):
    img = img.clone()
    height, width = img.shape
    half_window = window_size // 2
    objects = []
    locations = []
    depths = []
    ground = img.max()
    mask = img < (img.min() + 0.005)
    is_empty = mask.all()
    while not is_empty:
        h_i, w_i = mask.nonzero()[0]
        pp = cc_pix_avg(mask, h_i.item(), w_i.item())
        h_c, w_c = np.mean(pp, axis=0).round().astype(np.int)
        locations.append([h_c, w_c])
        # depths.append(img[int(h_c), int(w_c)].item())
        depths.append(img.min())
        h_c = np.clip(h_c, half_window, width-half_window)
        w_c = np.clip(w_c, half_window, width-half_window)
        objects.append(img[(h_c-half_window):(h_c+half_window), (w_c-half_window):(w_c+half_window)].clone())
        img[(h_c-half_window):(h_c+half_window), (w_c-half_window):(w_c+half_window)] = ground
        mask = img < (img.min()+0.005)
        is_empty = mask.all()
    if len(objects) > 0:
        objects = torch.stack(objects)
        locations = torch.tensor(locations)
        # sizes = torch.stack(sizes) * 3.47632
        depths = torch.tensor(depths)
    return objects, locations, depths


def kl_bernoulli(q, p):
    eps = 1e-20
    kl = q*torch.log(q+eps)+(1-q)*torch.log(1-q+eps)+q*torch.log((1-p)/(p+eps)+eps)
    return kl


def print_module(module, name, space):
    L = len(name)
    line = " "*space+"-"*(L+4)
    print(line)
    print(" "*space+"  "+name+"  ")
    print(line)
    module_str = module.__repr__()
    print("\n".join([" "*space+mstr for mstr in module_str.split("\n")]))
