"""Import mesh classes"""
from mesh import Mesh, Joint, Member, Support, Force
import numpy as np
from matplotlib import pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib import style
# define mandotary joints
JOINT1 = Joint(0, 0)
JOINT2 = Joint(3.5, 0)
JOINT3 = Joint(7, 0)
JOINT4 = Joint(10.5, 0)
JOINT5 = Joint(14, 0)

# price constants
MEMBER_COST = 15
JOINT_COST = 5

#
PIN_SUPPORT = Support.Base(True, True, True, True, True)
PIN_ROLLER_SUPPORT = Support.Base(False, False, True, True, False)


def triangle_mesh() -> Mesh:
    """
    Returns a traignular mesh in shape w 1 unit force on top straight down:
      ..o
     /.....\\
    o---o
    """
    j1 = Joint(0, 0)
    j2 = Joint(2, 0)
    j3 = Joint(1, 1)
    mem1 = Member(j1, j2)
    mem2 = Member(j2, j3)
    mem3 = Member(j3, j1)
    support1 = Support(j1, PIN_SUPPORT)
    support2 = Support(j2, PIN_ROLLER_SUPPORT)
    force1 = Force(j3, 2, 1.5)

    mesh = Mesh([mem1, mem2, mem3])
    mesh.add_support(support1)
    mesh.add_support(support2)
    mesh.apply_force(force1)

    return mesh


def quick_tutorial():
    """How to solve cost of skyciv truss."""

    # create mesh to house members and joints
    mesh = Mesh()

    # download member and node csv from skyciv then pass relative path to function
    mesh.from_csv("Nodes-3.csv", "Members-3.csv")

    # pass the prices of members and joints to get cost function
    cost = mesh.get_cost(MEMBER_COST, JOINT_COST)

    # print cost
    print(cost)

    return mesh


def main():
    """Main function."""

    mydict = {
        "a": 1,
        "b": 2
    }

    mesh = triangle_mesh()
    mesh.show()
    mesh.solve_supports()
    mesh.show()


if __name__ == "__main__":
    main()
