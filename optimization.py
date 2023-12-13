import random
import numpy as np
from src.trussty.mesh import Mesh, Joint, Member, Support, Force
import math

# plan


# make x amounts of base meshes
# put them in array

# in a loop

# use function to calculate fitness of each mesh incorporating price

# put fitness in a corresponding array


# use random library to calculate next batch based on fitness

# loop through new batch

# optimize each for an amount of iterations

# loop through joints and add probablity of adding a member

# re loop

# at some point scan for members crossing


import math


def ccw(A: Joint, B: Joint, C: Joint):
    return (C.y_coordinate-A.y_coordinate) * (B.x_coordinate-A.x_coordinate) > (B.y_coordinate-A.y_coordinate) * (C.x_coordinate-A.x_coordinate)

# Return true if line segments AB and CD intersect


def intersect(A: Joint, B: Joint, C: Joint, D: Joint):
    x1, x2, x3, x4 = A.x_coordinate, B.x_coordinate, C.x_coordinate, D.x_coordinate
    y1, y2, y3, y4 = A.y_coordinate, B.y_coordinate, C.y_coordinate, D.y_coordinate

    dx1 = x2 - x1
    dx2 = x4 - x3
    dy1 = y2 - y1
    dy2 = y4 - y3
    det = dx1 * dy2 - dx2 * dy1

    # parallel so must either be double member or not intersecting, small chance not but that is really small
    if det == 0:
        return None

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


print(intersect(Joint(-1, -1), Joint(1, 1), Joint(-1, 1), Joint(1, -1)))


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


def get_fitness(mesh: Mesh):
    cost = mesh.get_cost()
    num_joints = len(mesh.joints)
    num_members = len(mesh.members)
    num_supports = len(mesh.supports)

    # check for stable truss
    instability = (2*num_joints - (num_members+num_supports)
                   )**2 if (2*num_joints > num_members + num_supports) else 0

    # make sure members arent crossing, O(n^2) yikes
    intersections = 0
    for member1 in mesh.members:
        member1: Member
        for member2 in mesh.members:
            member2: Member
            intersect = intersect(
                member1.joint_a, member1.joint_b, member2.joint_a, member2.joint_b
            )
            if intersect:
                intersections += 1

    fitness = cost * (cost**instability) * (cost**intersections)

    return fitness


class Generation:
    probabilty_of_new_member = 0.1
    probabilty_of_new_joint = 0.1
    probabilty_of_delete_joint = 0.1

    max_loops = 10

    def __init__(self) -> None:
        self.__population: list[Mesh] = []
        self.__fitness: list[float] = []

    def reproduce(self):
        if len(self.__population) != len(self.__fitness):
            raise ValueError("Population and fitness not same size.")

        new_population = random.choices(
            self.__population, self.__fitness, k=len(self.__population))

        for mesh in new_population:
            # decide whether to add memeber
            add_member = False
            if random.random() < Generation.probabilty_of_new_member:
                add_member = True

            # decide whether to add member
            add_joint = False
            if random.random() < Generation.probabilty_of_new_joint:
                add_joint = True

            # decide whether to delete a joint
            delete_joint = False
            if random.random() < Generation.probabilty_of_delete_joint:
                delete_joint = True

            # if genetics wants to add member get indexs of joints tp connect it to
            if add_member:
                new_member_joint_a_index = 0
                new_member_joint_b_index = 0

                loops = 0
                while new_member_joint_a_index == new_member_joint_b_index:
                    new_member_joint_a_index = math.floor(
                        random.random() * len(mesh.joints))

                    new_member_joint_b_index = math.floor(
                        random.random() * len(mesh.joints))

                    loops += 1
                    if loops > Generation.max_loops:
                        add_member = False
                        break

            # if genetics wants to add joint get coordinates of new joint
            if add_joint:
                new_joint_x = random.random()*10*(-1 if random.random() > 0.5 else 1)
                new_joint_y = random.random()*10*(-1 if random.random() > 0.5 else 1)

                connected_to_a_index = math.floor(
                    random.random()*len(mesh.joints)
                )
                connected_to_b_index = math.floor(
                    random.random()*len(mesh.joints)
                )

            # if genetics wants to delete joint get coordinates index of joint to delete
            if delete_joint:
                delete_joint_index = 0

                loops = 0
                while delete_joint == new_member_joint_a_index or delete_joint == new_member_joint_b_index:
                    delete_joint_index = math.floor(
                        random.random()*len(mesh.joints))

                    loops += 1
                    if loops > Generation.max_loops:
                        delete_joint = False
                        break

            # get member
            new_member_joint_a: Joint = None
            new_member_joint_b: Joint = None
            connected_to_a: Joint = None
            connected_to_b: Joint = None
            for i, joint in mesh.joints:
                if add_member:
                    if i == new_member_joint_a_index:
                        new_member_joint_a = joint
                    if i == new_member_joint_b_index:
                        new_member_joint_b = joint

                if delete_joint:
                    if i == delete_joint_index:
                        mesh.delete_joint(joint)

                if add_joint:
                    if i == connected_to_a_index:
                        connected_to_a = joint
                    if i == connected_to_b_index:
                        connected_to_b = joint

            if add_joint:
                new_joint = Joint(
                    new_joint_x, new_joint_y, track_grad=True)
                mesh.add_member(Member(new_joint, connected_to_b))
                mesh.add_member((new_joint, connected_to_a))

            if add_member:
                mesh.add_member(Member(new_member_joint_a, new_member_joint_b))


class Evolve:
    def __init__(self) -> None:
        pass
