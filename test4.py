from pytruss import Mesh, Member, Joint, Force, Support
import random
import math
from matplotlib import pyplot as plt
import copy
import multiprocessing
import torch
import concurrent.futures as cf
import time
import numpy as np
MEMBER_COST = 15
JOINT_COST = 5


def make_mesh():
    J1 = Joint(0, 0, track_grad=False)
    J3 = Joint(3.5, 0, track_grad=False)
    J5 = Joint(7, 0, track_grad=False)
    J7 = Joint(10.5, 0, track_grad=False)
    J9 = Joint(14, 0, track_grad=False)

    pin_support = Support(J1, "p")
    roller_support = Support(J9, "rp")

    # top members
    mem1 = Member(J1, J3)
    mem2 = Member(J3, J5)
    mem3 = Member(J5, J7)
    mem4 = Member(J7, J9)

    mesh = Mesh([mem1, mem2, mem3, mem4])
    mesh.add_support(pin_support)
    mesh.add_support(roller_support)
    mesh.apply_force(Force(J1, 0, -7))
    mesh.apply_force(Force(J3, 0, -7))
    mesh.apply_force(Force(J5, 0, -7))
    mesh.apply_force(Force(J7, 0, -7))
    mesh.apply_force(Force(J9, 0, -7))

    joints = [
        J1,
        J3,
        J5,
        J7,
        J9,
    ]
    probs = [
        4,
        4,
        4,
        4,
        4,
    ]

    for _ in range(5):
        x = random.random()*20 - 10
        y = random.random()*20 - 10

        j = Joint(x, y, track_grad=True)

        ja = random.choices(joints, probs)[0]
        jb = random.choices(joints, probs)[0]
        while jb == ja:
            jb = random.choices(joints, probs)[0]

        mem = Member(j, jb)
        mem2 = Member(j, ja)
        mesh.add_member(mem)
        mesh.add_member(mem2)
        joints.append(j)
        probs.append(1)

    # mesh.show(show=True)

    return mesh


def optimizer(mesh: Mesh):
    mesh.optimize_cost(
        member_cost=MEMBER_COST,
        joint_cost=JOINT_COST,
        lr=1e-2,
        epochs=10000,
        optimizer=torch.optim.Adam,
        print_mesh=False,
        show_at_epoch=False,
        min_member_length=1,
        max_member_length=None,
        max_tensile_force=9,
        max_compresive_force=6,
        constriant_agression=2,
        progress_bar=False,
        show_metrics=False,
        update_metrics_interval=10,
    )


def main():

    start = time.time()

    meshes = [make_mesh() for _ in range(25)]

    with cf.ProcessPoolExecutor() as executor:
        training_tasks = [executor.submit(optimizer, mesh) for mesh in meshes]

        run = True
        plt.ion()
        fig, axs = plt.subplots(5, 5)

        axs = np.array(axs).flatten()
        while run:
            plt.tight_layout()
            for i, mesh in enumerate(meshes):
                axs[i].cla()
                mesh.show(ax=axs[i])

            tasks_running = 0
            for task in training_tasks:
                tasks_running += int(task.running())

            if tasks_running == 0:
                run = False

            plt.pause(0.01)

        plt.ioff()

    end = time.time()

    print(f"Took {end - start} seconds")


if __name__ == "__main__":
    main()

# processes: list[multiprocessing.Process] = []
# for mesh in meshes:
#     p = multiprocessing.Process(target=optimizer, args=[mesh])
#     p.start()
#     processes.append(p)

# for process in processes:
#     process.join()


# for mesh in meshes:
#     mesh.show()

# plt.show()
