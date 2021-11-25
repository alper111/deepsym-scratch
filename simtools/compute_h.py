import torch
import numpy as np

import comm


def computeH(im1Points, im2Points):
    '''
    Given a set of correspondences, computes the homogeneous transformation
    between these two sets. The returned matrix does the following transformation:

    im1Points = H, im2Points

    Note that H expects rows as features (pre-multiply with H).
    '''
    n = im1Points.shape[0]
    A = np.zeros((2*n, 9))
    for i in range(n):
        x_t = im1Points[i]
        y_t = im2Points[i]
        A[2*i, :] = np.concatenate([np.zeros(3), -x_t[2] * y_t, x_t[1] * y_t])
        A[2*i+1, :] = np.concatenate([x_t[2] * y_t, np.zeros(3), -x_t[0] * y_t])
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    return VT[-1].reshape(3, 3)


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


env = comm.Communication(20000)
env.open_connection()
env.stop()
env.start()

scales = np.linspace(1.0, 2.0, 10)
xrng = np.linspace(-0.4, -1.1, 5)[:4]
yrng = np.linspace(-0.35, 0.35, 5)[:4]
X_points = []
Y_points = []
H = None

for i in range(100):
    s = np.random.choice(scales)
    x = np.random.choice(xrng)
    y = np.random.choice(yrng)
    o = np.random.randint(0, 5)
    size = s * 0.1
    env.generate_object(o, [x, y, 0.7+size/2])
    env.set_object_scale(env.generated_objects[0], s, s, s)
    env.step()
    img = torch.tensor(env.get_depth())
    objs, locs, _ = find_objects(img, window_size=42)
    if len(objs) == 1:
        if H is not None:
            loc_est = torch.cat([locs.float(), torch.ones(locs.shape[0], 1)], dim=1)
            loc_est = torch.matmul(loc_est, H.T)
            loc_est = loc_est / loc_est[:, 2].reshape(-1, 1)
            print(f"Pixel: ({locs[0, 0]}, {locs[0, 1]}), Real: ({x:.3f}, {y:.3f}), Estimated: ({loc_est[0, 0]:.3f}, {loc_est[0, 1]:.3f}), Error: ({abs(x-loc_est[0, 0]):.3f}, {abs(y-loc_est[0, 1]):.3f})")
        else:
            print(f"Pixel: ({locs[0, 0]}, {locs[0, 1]}), Real: ({x:.3f}, {y:.3f}), Estimated: waiting...")
    X_points.append([locs[0, 0], locs[0, 1], 1])
    Y_points.append([x, y, 1])
    env.remove_object(env.generated_objects[0])
    env.step()

    if len(X_points) > 6:
        H = torch.tensor(computeH(np.array(Y_points), np.array(X_points)), dtype=torch.float)

H = torch.tensor(computeH(np.array(Y_points), np.array(X_points)), dtype=torch.float)
torch.save(H, "H.pt")
