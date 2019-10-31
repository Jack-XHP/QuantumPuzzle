from dwave_qbsolv import QBSolv
import numpy as np

# xyz =min_w wx + M(yz - 2(y+z)w + 3w)
# https://docs.dwavesys.com/docs/latest/c_handbook_3.html#reduction-by-substitution
def reduceBySubstitution(J, J_3D, M):
    ancil_var = -1
    for term in J_3D:
        weight = J_3D[term]
        weight_P = weight * M
        J_tmp = {(ancil_var, term[0]):weight,
                 (term[1], term[2]): weight_P,
                 (ancil_var, term[1]): -2 * weight_P,
                 (ancil_var, term[2]): -2 * weight_P,
                 (ancil_var, ancil_var): 3 * weight_P}
        for i in J_tmp:
            if i in J:
                J[i] += J_tmp[i]
            else:
                J[i] = J_tmp[i]
        ancil_var -= 1


# scale*center*(sum(neighbor) - target)^2
def sumToN(center, neighbor, target, J, J_3D, scale=1):
    for ele in neighbor:
        if center < ele:
            term = (center, ele)
        else:
            term = (ele, center)
        # for binary variable a^2 = a, thus a^2 - 2*target*a = -(2*target -1)a
        if term in J:
            J[term] -= (2 * target - 1) * scale
        else:
            J[term] = -(2 * target - 1) * scale
    for ele1 in neighbor:
        for ele2 in neighbor:
            if ele1 == ele2:
                continue
            else:
                if ele1 > ele2:
                    continue
                else:
                    weight = 2 * scale
                    term = [center, ele1, ele2]
                    term.sort()
                    term = tuple(term)
                    if term in J_3D:
                        J_3D[term] += weight
                    else:
                        J_3D[term] = weight


#scale*(sum(neighbor) - target)^2
def sumToN2(neighbor, target, J, scale=1):
    for ele1 in neighbor:
        for ele2 in neighbor:
            term = (ele1, ele2)
            if ele1 == ele2:
                # for binary variable a^2 = a, thus a^2 - 2*target*a = -(2*target -1)a
                weight = -2*target + 1
            elif ele1 > ele2:
                continue
            else:
                # 2ab term
                weight = 2
            if term in J:
                J[term] += weight * scale
            else:
                J[term] = weight * scale


def sumLessOne(neighbor, J, scale=1):
    for ele1 in neighbor:
        for ele2 in neighbor:
            term = (ele1, ele2)
            if ele1 < ele2:
                if term in J:
                    J[term] += scale
                else:
                    J[term] = scale


def oneLayer(origin, J,J_3D, grid):
    hight, width = grid.shape
    origin_var = [grid[index] for index in origin]
    all_var = grid.flatten()
    min_path_len = abs(origin[0][0] - origin[1][0]) + abs(origin[0][1] - origin[1][1])
    sumToN2(all_var, min_path_len, J, scale=1.2)
    for x in all_var:
        term = (x, x)
        if x in origin_var:
            weight = -hight*width * 1000
        else:
            weight = 0
        if term in J:
            J[term] += weight
        else:
            J[term] = weight
        i = (x-grid[0,0]) // width
        j = (x-grid[0,0]) % width
        near = []
        if j > 0:
            neighbor = grid[i][j - 1]
            near.append(neighbor)
        if j < width - 1:
            neighbor = grid[i][j + 1]
            near.append(neighbor)
        if i > 0:
            neighbor = grid[i - 1][j]
            near.append(neighbor)
        if i < hight - 1:
            neighbor = grid[i + 1][j]
            near.append(neighbor)
        if x in origin_var:
            sumToN(x, near, 1, J,J_3D, scale=10)
        else:
            sumToN(x, near, 2, J,J_3D, scale=1)


# 2 degree term
J = {}
# 3 degree term
J_3D = {}

depth, hight, width = (2,6,6)
grid = np.arange(depth*hight*width).reshape((depth, hight, width))

origin = (
    ((0,0), (5,5)),
    ((1,3), (3,3)),
)
origin_list = [i for i in origin]
origin_list = [i for i in origin_list]
for i in range(depth):
    oneLayer(origin[i], J, J_3D, grid[i])

if depth > 1:
    for x in range(hight):
        for y in range(width):
            same_bis = grid[:, x, y].flatten()
            if (x, y) in origin_list:
                scale = hight * width * 1000
            else:
                scale = 6
            sumLessOne(same_bis, J, scale=scale)


# convert all 3 degree term to qubo form
reduceBySubstitution(J, J_3D, 2)


print(J)
res = QBSolv().sample_qubo(J, num_repeats=1000)
samples = list(res.samples())
energy = list(res.data_vectors['energy'])
print(samples)
print(energy)
for i in range(len(samples)):
    result = samples[i]
    output = np.zeros((hight, width))
    # ignore ancillary variables, which are all negative, only get positive bits
    for l in range(depth):
        for x in range(hight):
            for y in range(width):
                bit = result[grid[l, x, y]] * (l+1)
                if bit != 0:
                    old_state = output[x, y]
                    if old_state != 0:
                        output[x,y] = old_state * 10 + bit
                    else:
                        output[x, y] = bit
    for l in range(depth):
        for ori in origin[l]:
            old_state = output[ori]
            output[ori] = old_state + 0.1 * (l+1)
    print("energy: {}_____________________________".format(energy[i]))
    print(output)
