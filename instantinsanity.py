import numpy as np
import networkx as nx
import minorminer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave_qbsolv import QBSolv


# scale*(sum(neighbor) - target)^2
def sumToN2(neighbor, target, Q, scale=1):

    for ele1 in neighbor:
        for ele2 in neighbor:
            term = (ele1, ele2)
            if ele1 == ele2:
                # for binary variable a^2 = a, thus a^2 - 2*target*a = -(2*target -1)a
                weight = -2 * target + 1
            elif ele1 > ele2:
                continue
            else:
                # 2ab term
                weight = 2
            if term in Q:
                Q[term] += weight * scale
            else:
                Q[term] = weight * scale


# including an edge should force inclusion of vertices
def edge_vertex_inclusion(edge, vertices, Q, scale=1):

    vertices = np.unique(vertices)  # some edges are to the same colour, we don't want to double count them
    for v in vertices:
        term = (edge, v)
        if term in Q:
            Q[term] += scale
        else:
            Q[term] = scale


# including an edge in subgraph A should exclude it from subgraph B
def ab_exclusion(n, Q, w_ab=1):

    for i in range(n):

        for j in range(0, 2 * n_edges, 2):

            idx = j + i * offset
            term = (idx, idx + 1)
            if term in Q:
                Q[term] += w_ab
            else:
                Q[term] = w_ab


def plot_problem(cubes, edges):

    w = 1
    h = 1
    coords = {'left': (0, 3), 'right': (2, 3), 'back': (3, 3), 'front': (1, 3), 'bottom': (1, 2), 'top': (1, 4)}
    x_max = 9
    x_min = 6
    y_min = 2.5
    y_max = 4.5
    colours = {'r': (x_min, y_max, '#FF4365'), 'g': (x_min, y_min, '#AACF90'), 'b': (x_max, y_min, '#8FAAD9'), 'y': (x_max, y_max, '#EAC435')}
    off = 0.3
    annotes = {'r': (6 - 3*off, 5), 'g': (6 - 3*off, 2 - off), 'b': (9 + off, 2 - off), 'y': (9 + off, 5)}

    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))

    for i, cube in enumerate(cubes):
        sides = []
        k = i % 2
        j = int((i > 1))
        ax = axs1[j][k]
        for key in cube.keys():
            rect = Rectangle(coords[key], width=w, height=h, facecolor=colours[cube[key]][2], alpha=0.6, edgecolor='k')
            ax.add_patch(rect)
            sides.append(rect)

        for edge in edges:
            (e1, e2) = edge
            e1 = cube[e1]
            e2 = cube[e2]
            ax = graph_plot(ax, e1, e2)

        for colour in colours.keys():
            (x, y, c) = colours[colour]
            ax.scatter(x, y, marker='o', s=256, c=c, edgecolors='k', zorder=5)

        ax.set_ylim((0.5, 6.5))
        ax.set_xlim((-1, 11))
        ax.title.set_text('Cube {}'.format(i+1))
        ax.axis('off')

    fig1.tight_layout()
    fig1.savefig('result/instantinsanity/problem1.png', dpi=400)
    fig1.show()

    return fig1, axs1


def plot_solution(soln, save=None):

    lims = (0, 3, 2.5, 4.5)
    (x_min, x_max, y_min, y_max) = lims
    off = 0.3
    colours = {'r': (x_min, y_max, '#FF4365'), 'g': (x_min, y_min, '#AACF90'), 'b': (x_max, y_min, '#8FAAD9'), 'y': (x_max, y_max, '#EAC435')}
    annotes = {'r': (x_min - 2 * off, y_max), 'g': (x_min - 2 * off, y_min - off), 'b': (x_max + off, y_min - off), 'y': (x_max + off, y_max)}

    edge_idxs = np.where(soln == 1)[0]
    w = 0.9
    h = 0.9
    left = []
    right = []
    front = []
    back = []
    sides = [0, left, right, back, front]

    plt.rcParams.update({'font.size': 30})
    fig2, axs2 = plt.subplots(1, 2, figsize=(20, 6))

    for idx in edge_idxs:
        cube = cubes[edge_mappings[idx][0]]
        edge = edge_mappings[idx][1]
        (c1, c2) = (cube[edge[0]], cube[edge[1]])
        if idx % 2 == 0:
            if (c1 not in left and c2 not in right) or len(left) == 0:
                left.append(c1)
                right.append(c2)
            elif c2 not in left and c1 not in right:
                left.append(c2)
                right.append(c1)
            else:
                left.append(c2)
                right.append(c1)
                print('Solution is incorrect.')
                #return edge_mappings[idx][0], None, None, None
            graph_plot(axs2[0], c1, c2, lims=lims, lbl_edge=str(edge_mappings[idx][0] + 1))
        else:
            if (c1 not in back and c2 not in front) or len(back) == 0:
                back.append(c1)
                front.append(c2)
            elif c2 not in back and c1 not in front:
                back.append(c2)
                front.append(c1)
            else:
                back.append(c2)
                front.append(c1)
                print('Solution is incorrect.')
                #return edge_mappings[idx][0], None, None, None
            graph_plot(axs2[1], c1, c2, lims=lims, lbl_edge=str(edge_mappings[idx][0] + 1))

    lbls = ['Subgraph A - Left/Right', 'Subgraph B - Front/Back']
    for i, ax in enumerate(axs2):

        for colour in colours.keys():
            (x, y, c) = colours[colour]
            ax.scatter(x, y, marker='o', s=1024, c=c, edgecolors='k', zorder=5)
            ax.annotate(colour.upper(), (x, y), annotes[colour])

        title = lbls[i]
        ax.set_ylim((2, 6))
        ax.set_xlim((-1, 4))
        ax.title.set_text(title)
        ax.axis('off')

    plt.tight_layout()
    if save is not None:
        fname = save + '_graphs.png'
        fig2.savefig(fname, dpi=400)
    fig2.show()

    lbls = ['', 'Left', 'Right', 'Back', 'Front']
    fig3, axs3 = plt.subplots(1, 5, figsize=(20, 5))

    for i, ax in enumerate(axs3):
        title = lbls[i]
        if i == 0:
            for j in range(len(sides[1])):
                xy = (1, len(sides) - j - 2)
                xytxt = (xy[0], xy[1] + 0.2)
                ax.annotate('Cube {}'.format(j + 1), xy, xytxt)
        else:
            for j, c in enumerate(sides[i]):
                xy = (1, len(sides[i]) - j - 1)
                rect = Rectangle(xy, width=w, height=h, facecolor=colours[c][2], alpha=0.6, edgecolor='k')
                ax.add_patch(rect)

        ax.set_ylim((0, 4.5))
        if i == 0:
            ax.set_xlim((0, 1))
        else:
            ax.set_xlim((0, 3))
            ax.title.set_text(title)
        ax.axis('off')

    plt.tight_layout()
    if save is not None:
        fname = save + '.png'
        fig3.savefig(fname, dpi=400)
    fig3.show()

    return fig2, axs2, fig3, axs3


def graph_plot(axes, e1, e2, lims=(6, 9, 2.5, 4.5), lbl_edge=None):

    (x_min, x_max, y_min, y_max) = lims

    colours = {'r': (x_min, y_max, '#FF4365'), 'g': (x_min, y_min, '#AACF90'), 'b': (x_max, y_min, '#8FAAD9'),
               'y': (x_max, y_max, '#EAC435')}
    (x1, y1, _) = colours[e1]
    (x2, y2, _) = colours[e2]
    if e1 == e2:  # edge from node to itself
        r = 0.5
        xy = np.sqrt(r ** 2 / 2) + 0.1
        if x1 == x_max and y1 == y_max:
            offset = (0, xy)
        elif x1 == x_max:
            offset = (0, -xy)
        elif y1 == y_max:
            offset = (0, xy)
        else:
            offset = (0, -xy)

        center = (x1 + offset[0], y1 + offset[1])
        loop = Circle(center, r, facecolor=None, lw=2, fill=False, edgecolor='k', zorder=0.1)
        axes.add_patch(loop)
        if lbl_edge:
            axes.annotate(lbl_edge, (center[0] - 1.1 * r, center[1]))
    else:
        x = [x1, x2]
        y = [y1, y2]
        axes.plot(x, y, c='k', lw=2, zorder=0.1)
        if lbl_edge:
            if x1 != x2:
                m = (y2 - y1)/(x2 - x1)
                p1 = x_min + 0.5
                p2 = m * (p1 - x1) + y1 + 0.1
                xy = (p1, p2)
            else:
                xy = (x1 - 0.2, y_min + 0.7)
            axes.annotate(lbl_edge, xy)

    return axes


def check_constraints(proposal, save=None, plot_soln=False):
    """
    Given a solution (array of binary variables), returns dictionary containing counts of constraint violations
    :param proposal: array of binary variables proposed as a solution to the problem instance
    :return violations: dictionary containing constraint violation counts
    """

    violations = {}
    proposal_2D = proposal.reshape((n, offset))
    edges_r = proposal_2D[:, 0:6]
    colours_r = proposal_2D[:, 6:]
    colour_counts = np.zeros(proposal.shape)

    # edge/colour indices for subgraphs
    Ae = np.arange(0, n_sides, 2)
    Be = np.arange(1, n_sides, 2)

    # subgraph arrays
    subgraphAe = proposal_2D[:, Ae]
    subgraphBe = proposal_2D[:, Be]

    # edge in subgraph A should exclude it from subgraph B
    ab_count = np.count_nonzero((subgraphAe * subgraphBe))
    violations['total'] = ab_count
    violations['A/B exclusion'] = ab_count

    # edge inclusion implies vertex inclusion
    in_count = 0
    for edge_idx in edge_mappings.keys():
        if proposal[edge_idx]:
            c_idxs = edge_mappings[edge_idx][2]
            for c_idx in c_idxs:
                colour_counts[c_idx] += proposal[c_idx]
            in_count += np.count_nonzero((proposal[c_idxs] != 1))
    violations['total'] += in_count
    violations['edge/vertex inclusion'] = in_count

    # each subgraph has all four colours each with degree 2
    colour_counts = colour_counts.reshape((n, offset))[:, 6:]
    c_count = np.count_nonzero((np.sum(colour_counts, axis=0) != 2))
    violations['total'] += c_count
    violations['colour'] = c_count

    # one and only one edge from each cube in a subgraph
    e_count = np.count_nonzero((np.sum(subgraphAe, axis=1) != 1)) + np.count_nonzero((np.sum(subgraphBe, axis=1) != 1))
    violations['total'] += e_count
    violations['edge'] = e_count



    # plot solution if problem instance has n = 4, and no violations
    soln = np.concatenate((edges_r, colours_r * 0), axis=1).flatten()
    if n == 4 and violations['total'] == 0 and plot_soln:
        _, _, _, _ = plot_solution(soln, save=save)

    return violations


if __name__ == "__main__":

    # initialize qubo problem
    Q = {}

    plt.rcParams.update({'font.size': 22})

    # Define problem
    n = 4  # number of cubes/colours
    n_sides = 6
    n_edges = n_sides // 2
    offset = 2 * n + n_sides  # index offset between cubes i.e. num bits / cube
    n_bits = (2 * n + n_sides) * n  # number of bits for the whole problem

    colour = np.arange(n)
    cube = np.arange(n)

    colours = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
    edges = [('right', 'left'), ('front', 'back'), ('bottom', 'top')]
    cube_sides = ['left', 'right', 'back',  'front', 'bottom', 'top']

    # cube definitions
    cube0 = {'left': 'r', 'right': 'g', 'back': 'b', 'front': 'y', 'bottom': 'r', 'top': 'r'}
    cube1 = {'left': 'r', 'right': 'b', 'back': 'g', 'front': 'y', 'bottom': 'y', 'top': 'r'}
    cube2 = {'left': 'b', 'right': 'r', 'back': 'y', 'front': 'b', 'bottom': 'g', 'top': 'g'}
    cube3 = {'left': 'g', 'right': 'r', 'back': 'g', 'front': 'y', 'bottom': 'y', 'top': 'b'}
    edge_mappings = {}
    cubes = [cube0, cube1, cube2, cube3]
    if n == 3:
        _, _ = plot_problem(cubes, edges)

    # use qpu or not
    use_qpu = True
    use_best = True

    if use_qpu or not use_best:
        # penalty weights
        w_ev = -1  # edge inclusion implies vertex inclusion
        w_ab = 10  # edge in subgraph A should exclude it from subgraph B
        w_colour = 1  # each subgraph has all four colours
        w_cube = 2  # one and only one edge from each cube in a subgraph
    else:
        (w_ev, w_ab, w_colour, w_cube) = (-1, 10, 1, 2)

    # Build QUBO

    for i in range(n_bits):
        term = (i, i)
        if i % offset > 5:
            Q[term] = 0  # biases are 0
        else:
            Q[term] = 0  # edge biases are zero

    # CONSTRAINT: edge inclusion implies vertex inclusion
    for i, cube in enumerate(cubes):

        for j, edge in enumerate(edges):

            # subgraph A
            edge_idx = 2 * j + i * offset  # index of the current edge node (for subgraph A i.e. even indices)
            vertices = np.array([n_sides + 2 * colours[cube[v]] + i * offset for v in edge])
            edge_vertex_inclusion(edge_idx, vertices, Q, scale=w_ev)
            edge_mappings[edge_idx] = [i, edge, np.copy(vertices)]

            # subgraph B - just odd indices i.e. increment by 1
            edge_idx += 1
            vertices += 1
            edge_vertex_inclusion(edge_idx, vertices, Q, scale=w_ev)
            edge_mappings[edge_idx] = [i, edge, np.copy(vertices)]

    # CONSTRAINT: including an edge in subgraph A should exclude it from subgraph B
    ab_exclusion(n, Q, w_ab=w_ab)

    # CONSTRAINT: All n colours in each subgraph, each colour has two edges
    target = 2

    for c in colours.values():

        # subgraph A
        neighbours = np.array([6 + 2 * c + cube * offset for cube in range(n)])
        sumToN2(neighbours, target, Q, scale=w_colour)

        # subgraph B
        neighbours += 1
        sumToN2(neighbours, target, Q, scale=w_colour)

    # CONSTRAINT: one and only one edge from each cube in a subgraph => sum over edges in a cube = 2
    for cube in range(n):

        # subgraph A
        target = 1
        neighbours = np.array([i + cube * offset for i in range(0, 2 * n_edges, 2)])
        sumToN2(neighbours, target, Q, scale=w_cube)

        # subgraph B
        neighbours += 1
        sumToN2(neighbours, target, Q, scale=w_cube)

        # # sum over edges
        target = 2
        neighbours = np.array([i for i in range(0, 2 * n_edges)])
        sumToN2(neighbours, target, Q, scale=w_cube)

    if use_qpu:
        print('Sampling with QPU...')
        sampler = 'qpu'
        solver_limit = 56
        G = nx.complete_graph(solver_limit)
        system = DWaveSampler()
        embedding = minorminer.find_embedding(Q.keys(), system.edgelist)
        #print(embedding)
        res = QBSolv().sample_qubo(Q,
                                   solver=FixedEmbeddingComposite(system, embedding),
                                   solver_limit=solver_limit,
                                   num_reads=1000)

    else:
        print('Sampling with classical solver...')
        sampler = 'sim'
        res = QBSolv().sample_qubo(Q, num_repeats=10)

    # check constraints
    samples = list(res.samples())
    energy = list(res.data_vectors['energy'])

    n_print = 20
    energies = []
    n_violations = []
    n_occ = []
    for num, sample in enumerate(res.data(fields=['sample', 'energy', 'num_occurrences'], sorted_by='energy')):
        if num < n_print:
            sample_arr = np.array([sample.sample[key] for key in sample.sample.keys()])
            violations = check_constraints(sample_arr,
                                           save='result/instantinsanity/nonsoln_problem{0}{1}'.format(num, sampler),
                                           plot_soln=True)
            energy = sample.energy

            if num ==0:
                n_occ.append(sample.num_occurrences)
            elif energy == energies[-1]:
                n_occ[-1] += sample.num_occurrences
            else:
                n_occ.append(sample.num_occurrences)

            energies.append(energy)
            n_violations.append(violations['total'])
            print('Energy = {}'.format(sample.energy))
            print('constraint violations = {}'.format(violations))
            print('=======================================================================================')


    res_fig = plt.figure(figsize=(10,10))
    x = np.unique(energies)
    plt.bar(x, n_occ, color='#8FAAD9')
    plt.xlabel('energy')
    plt.ylabel('number of samples')
    plt.xticks(x, x)
    plt.tight_layout()
    plt.savefig('result/instantinsanity/sample_hist{}.png'.format(sampler), dpi=400)
    plt.show()
    print('Energies = {}'.format(energies))

    res_fig = plt.figure(figsize=(10,10))
    plt.scatter(n_violations, energies)
    plt.xlabel('constraint violations')
    plt.ylabel('energy')
    plt.tight_layout()
    plt.savefig('result/instantinsanity/energy_levels{}.png'.format(sampler), dpi=400)
    plt.show()
    print('Energies = {}'.format(energies))



