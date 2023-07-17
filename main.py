"""Import mesh classes"""
from mesh import Mesh, Joint, Member, Support, Force

# define mandotary joints
JOINT1 = Joint(0, 0)
JOINT2 = Joint(3.5, 0)
JOINT3 = Joint(7, 0)
JOINT4 = Joint(10.5, 0)
JOINT5 = Joint(14, 0)

# price constants
MEMBER_COST = 15
JOINT_COST = 5


def quick_tutorial():
    """How to solve cost of skyciv truss."""

    # create mesh to house members and joints
    mesh = Mesh()

    # download member and node csv from skyciv then pass relative path to function
    mesh.from_csv("Nodes-2.csv", "Members-2.csv")

    # pass the prices of members and joints to get cost function
    cost = mesh.get_cost(MEMBER_COST, JOINT_COST)

    # print cost
    print(cost)


def main():
    """Main function."""
    quick_tutorial()


if __name__ == "__main__":
    main()
