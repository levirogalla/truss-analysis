"""Import mesh classes"""
from src.trussty.mesh import Mesh, Joint, Member, Support, Force
import numpy as np
from matplotlib import pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib import style
import torch
from tqdm import tqdm
# define mandotary joints
JOINT1 = Joint(0, 0)
JOINT2 = Joint(3.5, 0)
JOINT3 = Joint(7, 0)
JOINT4 = Joint(10.5, 0)
JOINT5 = Joint(14, 0)

# price constants
MEMBER_COST = 15
JOINT_COST = 5


def triangle_mesh() -> Mesh:
    """
    Returns a traignular mesh in shape w 1 unit force on top straight down:
      ..o
     /.....\\
    o---o
    """
    j1 = Joint(0, 0)
    j2 = Joint(2, 0)
    j3 = Joint(0.5, 1, track_grad=True)
    # j4 = Joint(0, 1, track_grad=True)
    mem1 = Member(j1, j2)
    mem2 = Member(j2, j3)
    mem3 = Member(j3, j1)
    # mem4 = Member(j4, j3)
    # mem5 = Member(j4, j1)
    support1 = Support(j1, "p")
    support2 = Support(j2, "rp")
    force1 = Force(j3, 0, -2)
    mesh = Mesh([mem1, mem2, mem3])

    mesh.add_support(support1)
    mesh.add_support(support2)
    mesh.apply_force(force1)

    return mesh


def square_mesh() -> Mesh:
    j1 = Joint(0, 0)
    j2 = Joint(1, 0)
    j3 = Joint(100, 100, track_grad=True)
    j4 = Joint(0, -30, track_grad=True)

    bottom = Member(j1, j2)
    right = Member(j2, j3)
    top = Member(j3, j4)
    left = Member(j4, j1)
    diag = Member(j1, j3)

    mesh = Mesh([bottom, right, top, left, diag])
    mesh.add_support(Support(j1, "p"))
    mesh.add_support(Support(j2, "rp"))
    mesh.apply_force(Force(j3, 1, -1))

    return mesh, j3


def quick_tutorial():
    """How to solve cost of skyciv truss."""

    # create mesh to house members and joints
    mesh = Mesh()

    # download member and node csv from skyciv then pass relative path to function
    mesh.from_csv("Nodes-4.csv", "Members-4.csv")

    # add supports
    mesh.add_support(Support(Joint(0, 0), "p"))
    mesh.add_support(Support(Joint(14, 0), "rp"))

    # add forces
    for i in range(5):
        mesh.apply_force(Force(Joint(i*3.5, 0), 0, -7))

    # pass the prices of members and joints to get cost function
    cost = mesh.get_cost(MEMBER_COST, JOINT_COST)

    # print cost
    print(cost)

    # show the truss
    mesh.show()

    return mesh


def main():
    """Main function."""

    mesh, j3 = square_mesh()
    mesh: Mesh
    j3: Joint
    original_cost = mesh.get_cost(MEMBER_COST, JOINT_COST)

    mesh.solve_supports()
    mesh.solve_members()
    mesh.print_members()
    mesh.show(show=True)

    mesh.optimize_cost(
        MEMBER_COST,
        JOINT_COST,
        0.01,
        100000,
        optimizer=torch.optim.Adam,
        print_mesh=False,
        show_at_epoch=True,
        min_member_length=1,
        max_member_length=None,
        max_tensile_force=None,
        max_compresive_force=None,
        constriant_agression=1,
        progress_bar=True,
        show_metrics=True,
        update_metrics_interval=50,

    )

    mesh.print_members()
    mesh.solve_supports()
    mesh.solve_members()
    print()
    mesh.print_members()
    print(f"New cost: {mesh.get_cost(MEMBER_COST, JOINT_COST)}")
    print(f"Original cost: {original_cost}")
    mesh.show()


if __name__ == "__main__":
    main()
