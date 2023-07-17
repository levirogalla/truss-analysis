"""Import dataclass."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(unsafe_hash=True, init=False)
class Joint:
    """Defines a joint."""

    def __init__(self, x_coordinate: float, y_coordinate: float):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.__members: list["Member"] = []
        self.__forces: list["Force"] = []
        self.__vector = np.array([self.x_coordinate, self.y_coordinate])
        self.__support: Support.Base = None

    def __eq__(self, __value: "Joint") -> bool:
        if (self.x_coordinate == __value.x_coordinate and self.y_coordinate == __value.y_coordinate):
            return True
        return False

    def __setattr__(self, __name: str, __value: Any) -> None:
        # to avoid calling these varible during creation when they dont exist yet
        if len(self.__dict__) == 6:
            if __name == "x_coordinate":
                print("Can not change joint attributes")
            if __name == "y_coordinate":
                print("Can not change joint attributes")

        else:
            super().__setattr__(__name, __value)

    def add_support(self, support: "Support.Base") -> None:
        """Add support to joint."""
        self.__support = support

    def get_support(self) -> "Support.Base":
        """Get support on joint."""
        return self.__support

    def add_member(self, member: "Member") -> None:
        """Adds a member to joint."""
        self.__members.append(member)

    def get_members(self) -> list["Member"]:
        """Get members attached to joint."""
        return self.__members

    def apply_force(self, force: "Force") -> None:
        """Adds force to joint."""
        self.__forces.append(force)

    def get_forces(self) -> list["Force"]:
        """Get forces on a joint."""
        return self.__forces

    def get_vector(self):
        """Gets joint position in vector representation."""
        return self.__vector

    def __repr__(self) -> str:
        return f"{self.x_coordinate}, {self.y_coordinate}"


@dataclass(frozen=True)
class Member:
    """Defines a member."""
    joint_a: Joint
    joint_b: Joint
    force: float = 0
    force_type: str = ""

    @property
    def len(self) -> float:
        "Returns length of member."
        diff = self.joint_a.get_vector() - self.joint_b.get_vector()
        length = np.linalg.norm(diff)
        return length


@dataclass(init=False)
class Force:
    """Defines a force on a joint."""

    def __init__(self, joint: Joint, x_component: float, y_component: float) -> None:
        self.joint = joint
        self.x_component = x_component
        self.y_component = y_component
        self.__vector = np.array(
            [self.x_component, self.y_component], dtype=np.float32
        )

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)

        if len(self.__dict__) == 4:
            self.__vector[0] = self.x_component
            self.__vector[1] = self.y_component

    @property
    def magnitude(self) -> float:
        """Return magnitude of force vector."""
        mag = np.linalg.norm(self.__vector)
        return mag

    def get_vector(self):
        "Gets vector representation of force."
        return self.__vector


@dataclass
class Support:
    """Defines a support for a joint."""
    joint: Joint
    base: "Base"

    @dataclass
    class Base:
        "Create a base support"
        support_force_positve_x: bool
        support_force_negative_x: bool
        support_force_positve_y: bool
        support_force_negative_y: bool
        support_moment: bool

    def __post_init__(self) -> None:
        self.x_reaction = 0
        self.y_reaction = 0
        self.moment_reaction = 0


class Mesh:
    """Define a mesh with joints and member"""

    def __init__(self, members: list[Member] = None) -> None:
        self.joints: set[Joint] = set()
        self.members: list[Member] = list() if members is None else members
        self.forces: list[Force] = list()
        self.supports: list[Support] = list()

        for member in self.members:
            self.add_member(member)

    def print(self) -> None:
        """Prints mesh to terminal."""
        for member in self.members:
            print(f"{member.joint_a} ---- {member.joint_b} | {member.len}")

    def add_member(self, member: Member) -> None:
        """Adds a member to instance."""

        # adds joints to mesh
        self.joints.add(member.joint_a)
        self.joints.add(member.joint_b)

        # adds member to joints
        member.joint_a.add_member(member)
        member.joint_b.add_member(member)

        # add member to mesh
        self.members.append(member)

    def add_support(self, support: Support) -> None:
        """Adds support to mesh."""
        if support in self.supports:
            raise ValueError(f"{support} is in the mesh already.")

        # adds support to joint
        support.joint.add_support(support)

        # add support to mesh
        self.supports.append(support)

    def apply_force(self, force: Force) -> None:
        """Applies a force to a joint in the mesh"""
        # check if joint exists
        if force.joint not in self.joints:
            raise ValueError(f"{force.joint} is not in mesh joints.")

        # adds force to joint
        force.joint.apply_force(force)

        # adds force to mesh
        self.forces.append(force)

    def get_total_length(self) -> float:
        """Returns sum of the lenths of the member in the mesh."""
        total = 0
        for member in self.members:
            total += member.len
        return total

    def get_joints(self) -> list[Joint]:
        return self.joints

    def get_cost(self, member_cost: float, joint_cost: float) -> float:
        """
        Returns cost of mesh with parameter provided.
        member_cost: cost unit per distance unit.
        joint_cost: cost unit per joint.
        """
        cost = 0
        cost += (len(self.joints) * joint_cost)
        cost += (self.get_total_length() * member_cost)

        return cost

    def from_csv(self, path_to_node_csv: str, path_to_member_csv: str):
        """Import nodes and meshes from csv."""
        node_file = pd.read_csv(path_to_node_csv, index_col="Id")
        member_file = pd.read_csv(path_to_member_csv, index_col="Id")

        for _, row in member_file.iterrows():
            node_a = row["Node A"]
            node_b = row["Node B"]
            joint_a = Joint(
                node_file.loc[node_a, "X Position (m)"], node_file.loc[node_a, "Y Position (m)"])
            joint_b = Joint(
                node_file.loc[node_b, "X Position (m)"], node_file.loc[node_b, "Y Position (m)"])
            member = Member(joint_a, joint_b)
            self.add_member(member)

    def solve_supports(self):
        """Solve support reactions on mesh."""
        # move all forces to a point, picks random support joint to transfer to
        base_joint = self.supports[0].joint.get_vector()
        force_on_base_joint = np.zeros(2, dtype=np.float32)
        moment_about_base_joint: float = 0

        for force in self.forces:
            force_vector = force.get_vector()
            force_joint_vector = force.joint.get_vector()
            force_on_base_joint += force_vector

            distance_vector = force_joint_vector - base_joint
            moment = np.cross(force_vector, distance_vector)
            moment_about_base_joint += moment

        print(force_on_base_joint)
        print(moment_about_base_joint)

    def solve(self):
        """Solve for internal forces in mesh."""

        # loop through every joint in mesh and make equation for
        for joint in self.joints:
            pass

        pass
