"""Import dataclass."""
from dataclasses import dataclass
from typing import Any, Union
import torch
import pandas as pd
from matplotlib import pyplot as plt
from support_markers import TF, TP, RP, P, F
from tqdm import tqdm


@dataclass(unsafe_hash=True, init=False)
class Joint:
    """Defines a joint."""

    def __init__(self, x_coordinate: float, y_coordinate: float, track_grad=False):
        self.__x_coordinate = x_coordinate
        self.__y_coordinate = y_coordinate
        self.__members: list["Member"] = []
        self.__forces: list["Force"] = []
        self.__vector = torch.tensor(
            [self.__x_coordinate, self.__y_coordinate], dtype=torch.float32, requires_grad=track_grad)
        self.__support: Support.Base = None
        self.__track_grad = track_grad

    def __eq__(self, __value: "Joint") -> bool:
        if (self.__x_coordinate == __value.x_coordinate and self.__y_coordinate == __value.y_coordinate):
            return True
        return False

    @property
    def track_grad(self):
        return self.__track_grad

    @property
    def x_coordinate(self):
        """Get x coordinate of joint."""
        return self.__x_coordinate

    @property
    def y_coordinate(self):
        """Get y coordinate of joint."""
        return self.__y_coordinate

    def set_x(self, x_coordinate: float):
        """Setter for x."""
        self.__x_coordinate = x_coordinate
        self.__vector = torch.tensor(
            [x_coordinate, self.__y_coordinate], dtype=torch.float32, requires_grad=self.__track_grad)

    def set_y(self, y_coordinate: float):
        """Setter for y"""
        self.__y_coordinate = y_coordinate
        self.__vector = torch.tensor(
            [self.__x_coordinate, y_coordinate], dtype=torch.float32, requires_grad=self.__track_grad)

    def add_support(self, support: "Support.Base") -> None:
        """Add support to joint."""
        self.__support = support

    def sum_forces(self, force_type="all"):
        total_vec = torch.zeros(2, dtype=torch.float32)
        for force in self.__forces:
            if force.type == force_type:
                total_vec += force.vector
            elif force_type == "all":
                total_vec += force.vector
            else:
                raise ValueError(f"{force_type} is not a valid argument.")
        return Force(self, total_vec[0], total_vec[1], force_type=(force_type if force_type != "all" else "none"))

    @property
    def support(self) -> "Support.Base":
        """Get support on joint."""
        return self.__support

    def add_member(self, member: "Member") -> None:
        """Adds a member to joint."""
        self.__members.append(member)

    @property
    def members(self) -> list["Member"]:
        """Get members attached to joint."""
        return self.__members

    def apply_force(self, force: "Force") -> None:
        """Adds force to joint."""
        self.__forces.append(force)

    @property
    def forces(self) -> list["Force"]:
        """Get forces on a joint."""
        return self.__forces

    @property
    def vector(self):
        """Gets joint position in vector representation."""
        return self.__vector

    def __repr__(self) -> str:
        return f"({self.x_coordinate}, {self.y_coordinate})"


@dataclass
class Member:

    """Defines a member."""
    joint_a: Joint
    joint_b: Joint

    def __post_init__(self):
        self.__force: torch.Tensor = torch.tensor(
            0, dtype=torch.float32)
        self.__force_type: str = ""

    def __hash__(self) -> int:
        return hash(id(self))

    def set_force(self, force, force_type: str):
        """Set force. "c" for compresive, "t" for tensile."""
        if force_type not in ["c", "t"]:
            raise ValueError("Not valid force type.")
        self.__force = force
        self.__force_type = force_type

    def vector(self, vector_base: Joint = None):
        """Returns vector from joint a to b unless base is specified."""
        if vector_base is None or vector_base == self.joint_a:
            vector = self.joint_b.vector - self.joint_a.vector
        elif vector_base == self.joint_b:
            vector = self.joint_a.vector - self.joint_b.vector
        else:
            raise ValueError(f"{vector_base} does not belong to this member.")
        return vector

    @property
    def len(self) -> float:
        "Returns length of member."
        diff = self.joint_a.vector - self.joint_b.vector
        length = torch.norm(diff)
        return length

    @property
    def force(self):
        """Get force in the member."""
        return self.__force

    @property
    def force_type(self):
        """Get force type in the member."""
        return self.__force_type


@dataclass(init=False)
class Force:
    """Defines a force on a joint."""

    def __init__(self, joint: Joint, x_component: float, y_component: float, **kwargs) -> None:
        self.__joint = joint
        self.__x_component = x_component
        self.__y_component = y_component
        self.__vector = torch.tensor(
            [self.__x_component, self.__y_component], dtype=torch.float32)
        # only for internal use
        self.__type = "applied"

        for key, val in kwargs.items():
            if key == "force_type":
                if val == "applied":
                    self.__type = "applied"

                elif val == "reaction":
                    self.__type = "reaction"

                elif val == "none":
                    self.__type = "none"

                else:
                    raise ValueError(
                        f"{val} is not a valid argument for force_type. Valid arguments: 'applied', 'reaction', 'none'.")

    @property
    def joint(self):
        return self.__joint

    @property
    def x_component(self):
        return self.__x_component

    def set_x(self, value: float):
        """Set x component."""
        self.__x_component = value
        self.__vector[0] = value

    @property
    def y_component(self):
        return self.__y_component

    def set_y(self, value: float):
        """Set y component."""
        self.__y_component = value
        self.__vector[1] = value

    @property
    def magnitude(self) -> float:
        """Return magnitude of force vector."""
        mag = torch.norm(self.__vector)
        return mag

    @property
    def vector(self):
        "Gets vector representation of force."
        return self.__vector

    @property
    def type(self):
        """Returns force type: tension or compresion."""
        return self.__type


@dataclass
class Support:
    """
    Defines a support for a joint.
    Base types: "p"-pin, "r"-roller, "f"-fixed, "t"-track.
    """
    joint: Joint
    base: Union["Base", str]

    @dataclass(unsafe_hash=True)
    class Base:
        """
        Create a base support.
        """
        support_force_positve_x: bool
        support_force_negative_x: bool
        support_force_positve_y: bool
        support_force_negative_y: bool
        support_moment: bool

        @staticmethod
        def code_to_base(code: str):
            """Get support base from code."""
            codes = {
                "p": Support.Base(True, True, True, True, False),
                "f": Support.Base(True, True, True, True, True),
                "tf": Support.Base(False, False, True, True, True),
                "tp": Support.Base(False, False, True, True, False),
                "rp": Support.Base(False, False, True, False, False),

            }
            try:
                return codes[code]
            except KeyError as exc:
                raise KeyError(f"{code} is not a valid support type.") from exc

        @staticmethod
        def base_to_code(base: "Support.Base"):
            """Get code from base."""
            bases = {
                Support.Base(True, True, True, True, False): "p",
                Support.Base(True, True, True, True, True): "f",
                Support.Base(False, False, True, True, True): "tf",
                Support.Base(False, False, True, True, False): "tp",
                Support.Base(False, False, True, False, False): "rp",

            }
            try:
                return bases[base]
            except KeyError:
                return "custom"

    def __post_init__(self) -> None:
        self.x_reaction = 0
        self.y_reaction = 0
        self.moment_reaction = 0

        if isinstance(self.base, str):
            self.base = Support.Base.code_to_base(self.base)


class Mesh:
    """
    Define a mesh with joints and member.
    Do not delete any thing once its been added e.g. Mesh().forces.pop(index). This is not 
    supported yet.
    ONLY ADD STUFF WITH SETTER FUNCTIONS!
    """

    def __init__(self, members: list[Member] = None) -> None:
        self.__joints: set[Joint] = set()
        self.__members: dict[Member: int] = dict()
        self.__member_count: int = 0
        self.__forces: list[Force] = list()
        self.__supports: list[Support] = list()

        for member in members if members is not None else []:
            self.add_member(member)

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        # print(self.get_cost())

    def print(self) -> None:
        """Prints mesh to terminal."""
        for member in self.__members:
            print(f"{member.joint_a} ---- {member.joint_b} | {member.len}")

    @property
    def members(self) -> dict[Member: int]:
        return self.__members

    def add_member(self, member: Member) -> None:
        """
        Adds a member to instance.
        Will add joints connected to member implictly.
        """

        # adds joints to mesh
        self.__joints.add(member.joint_a)
        self.__joints.add(member.joint_b)

        # adds member to joints
        member.joint_a.add_member(member)
        member.joint_b.add_member(member)

        # add member to mesh pointing to its id, needed for indexing internal forces matrix
        self.__members[member] = self.__member_count
        self.__member_count += 1

    @property
    def supports(self) -> list[Support]:
        return self.__supports

    def add_support(self, support: Support) -> None:
        """
        Adds support to mesh.
        Will implicitly add support to joint.
        """
        if support in self.__supports:
            raise ValueError(f"{support} is in the mesh already.")

        if support.joint not in self.__joints:
            raise ValueError(f"{support} joint does not exist in mesh.")

        # adds support to joint
        support.joint.add_support(support)

        # add support to mesh
        self.__supports.append(support)

    @property
    def forces(self) -> list[Force]:
        return self.__forces

    def apply_force(self, force: Force) -> None:
        """
        Applies a force to a joint in the mesh.
        Will apply the force to the joint object implicitly.
        """
        # check if joint exists
        if force.joint not in self.__joints:
            raise ValueError(f"{force.joint} is not in mesh joints.")

        # adds force to joint
        force.joint.apply_force(force)

        # adds force to mesh
        self.__forces.append(force)

    def get_total_length(self) -> float:
        """Returns sum of the lenths of the member in the mesh."""
        total = 0
        for member in self.__members:
            total += member.len
        return total

    @property
    def joints(self) -> list[Joint]:
        """Returns joints"""
        return self.__joints

    def get_cost(self, member_cost: float, joint_cost: float) -> torch.Tensor:
        """
        Returns cost of mesh with parameter provided.
        member_cost: cost unit per distance unit.
        joint_cost: cost unit per joint.
        """
        cost = torch.tensor(0, dtype=torch.float32)
        cost += (len(self.__joints) * joint_cost)
        cost += (self.get_total_length() * member_cost)
        return cost

    def from_csv(self, path_to_node_csv: str, path_to_member_csv: str):
        """Import nodes and meshes from csv. Only supports skyciv format."""
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

    def show(self):
        """Show the visual truss"""
        joint_size = 5
        joint_color = "lightblue"

        member_width = 0.6
        member_color = "black"

        force_arrow_scale = 0.1
        force_arrow_head_width = 0.01
        applied_force_arrow_color = "darkblue"
        reaction_force_arrow_color = "darkred"

        support_size = joint_size*4
        support_color = "red"
        default_support_marker = "D"

        plt.xlim(-1, 3)
        plt.ylim(-1, 3)

        for support in self.__supports:
            support_marker = default_support_marker
            support_type = Support.Base.base_to_code(support.base)
            if support_type == "p":
                support_marker = P
            if support_type == "f":
                support_marker = F
            if support_type == "rp":
                support_marker = RP
            if support_type == "tp":
                support_marker = TP
            if support_type == "tf":
                support_marker = TF

            plt.plot(support.joint.x_coordinate,
                     support.joint.y_coordinate,
                     marker=support_marker, markersize=support_size,
                     color=support_color)

        for member in self.__members:
            x_values = [member.joint_a.x_coordinate,
                        member.joint_b.x_coordinate]
            y_values = [member.joint_a.y_coordinate,
                        member.joint_b.y_coordinate]
            plt.plot(x_values, y_values, 'o',
                     linestyle="-", color=member_color,
                     markerfacecolor=joint_color,
                     markeredgewidth=0.2,
                     markersize=joint_size,
                     linewidth=member_width
                     )

        for force in self.__forces:
            plt.arrow(force.joint.x_coordinate, force.joint.y_coordinate,
                      force.vector[0]*force_arrow_scale,
                      force.vector[1]*force_arrow_scale,
                      head_width=force_arrow_head_width,
                      color=(applied_force_arrow_color if force.type
                             == "applied" else reaction_force_arrow_color)
                      )

        plt.show()
    # assume right and up and all moments are positive

    def solve_supports(self, print_reactions: bool = False):
        """Solve support reactions on mesh."""

        # calculate total force
        force_on_base_joint = torch.zeros(2, dtype=torch.float32)
        for force in self.__forces:
            force_on_base_joint += force.vector

        # +2 for the two force equilbrium equations
        num_of_equations = (len(self.__supports)+2)

        # calculates number of variables/columns which is the number of supports *3 for x component forces, y
        # component forces and supported moment
        num_of_variables = len(self.__supports)*3

        # generates support matrix to be populated
        # row 0 will be x supports
        # row 1 will be y supports
        # column # will correspond to index of support in self.supports, even is x component, odd is y component
        support_matrix = torch.zeros(
            [num_of_equations, num_of_variables], dtype=torch.float32)
        # make it square
        # support_matrix = torch.zeros(
        #     [num_of_variables, num_of_variables], dtype=torch.float32)

        # eg: [
        # [coefFx1 = 1, coefFx2 = 1,           0,           0,                  0]
        # [          0,           0, coefFy1 = 1, coefFy2 = 1,                  0]
        # [ disFx1 = ?,      disFx2,  disFy1 = ?,  disFy2 = 1, supportsMoment = 1]
        # ]

        # agument vecotr to hold what the matrix should be equated to
        augment_vector = torch.zeros(num_of_equations, dtype=torch.float32)
        augment_vector[0] = -1*force_on_base_joint[0]
        augment_vector[1] = -1*force_on_base_joint[1]

        # iterate over supports
        for i, support in enumerate(self.__supports):
            # check to see if the support provides reaction x or y or m forces, negative x or postive x not supported yet
            if support.base.support_force_negative_x or support.base.support_force_positve_x:
                # assuming support is providing force in positive x
                support_matrix[0, i] = 1
            if support.base.support_force_negative_y or support.base.support_force_positve_y:
                # assuming support is providing force in posity y
                support_matrix[1, i + len(self.__supports)] = 1
            if support.base.support_moment:
                # assuming support is providing positive moment
                support_matrix[i+2, len(self.__supports)*2 + i] = 1

            # initialize base joint to find moment about it
            base_joint_vector = support.joint.vector
            moment_about_base_joint: float = 0
            for force in self.__forces:
                force_joint_vector = force.joint.vector
                distance_vector = force_joint_vector - base_joint_vector
                moment = torch.cross(
                    torch.cat((distance_vector, torch.tensor([0])), dim=0), torch.cat((force.vector, torch.tensor([0])), dim=0))
                moment_about_base_joint += moment[2]

            # negative because being moved to other side of equals sign
            augment_vector[i+2] = -1*moment_about_base_joint

            for j, other_support in enumerate(self.__supports):
                distance_vector = other_support.joint.vector - base_joint_vector
                x_distance = distance_vector[0]
                y_distance = distance_vector[1]
                # assumes support is providing force in the positive x
                x_force = int(
                    other_support.base.support_force_negative_x or
                    other_support.base.support_force_positve_x)
                # assumes support is providing force in the positive y
                y_force = int(
                    other_support.base.support_force_negative_y or
                    other_support.base.support_force_positve_y)

                support_matrix[i+2, j] = y_distance*x_force
                support_matrix[i+2, j+len(self.__supports)
                               ] = x_distance*y_force

        # is this the best way to do it?
        reactions = torch.linalg.lstsq(
            support_matrix, augment_vector).solution

        for i, support in enumerate(self.__supports):

            # add force data to support object (may make this hold a force object instead)
            support.x_reaction = reactions[i]
            support.y_reaction = reactions[i + len(self.__supports)]
            support.moment_reaction = reactions[i + 2*len(self.__supports)]

            # add force data to mesh object (will implicitly add to joint object)
            force = Force(
                support.joint, reactions[i],
                reactions[i + len(self.__supports)], force_type="reaction"
            )
            self.apply_force(force)

            if print_reactions:
                print(
                    f"""
For support at {support.joint}:
    x reaction: {support.x_reaction}
    y reaction: {support.y_reaction}
    moment reaction: {support.moment_reaction}""")

    def solve_members(self):
        """Solve for internal forces in mesh."""
        # loop through every joint in mesh and make equation for

        # ever member has an x component and y component
        num_of_variable = self.__member_count

        # finds the max number of equations, case where every node is connected
        # times 2 for x and y
        max_num_of_equations = len(self.__members)*len(self.__joints)*2
        forces_matrix = torch.zeros(
            [max_num_of_equations, num_of_variable], dtype=torch.float32)

        known_forces = torch.zeros(max_num_of_equations, dtype=torch.float32)

        # first have of the rows are x components and second half are y
        # self.members dict points to the index of
        r = 0
        for joint in self.__joints:
            force_on_joint = joint.sum_forces()
            for member in joint.members:
                c = self.__members[member]
                x_component_ratio = member.vector(
                    joint)[0]/torch.norm(member.vector())
                y_component_ratio = member.vector(
                    joint)[1]/torch.norm(member.vector())

                forces_matrix[r, c] = -1*x_component_ratio
                forces_matrix[r+1, c] = -1*y_component_ratio
                # *-1 because moving to other side of equalls
                known_forces[r] = -1*force_on_joint.x_component
                known_forces[r+1] = -1*force_on_joint.y_component

            r += 2

        member_forces = torch.linalg.lstsq(
            forces_matrix, known_forces).solution

        for member, local_id in self.__members.items():
            force = member_forces[local_id]
            if force > 0:
                force_type = "c"
            else:
                force_type = "t"

            member.set_force(torch.abs(force), force_type)

    def __optimize_member_length(self, joint: Joint, min_member_length=None, max_member_length=None):
        current_grad = joint.vector.grad.data
        for member in self.__members:
            if min_member_length is not None:
                if member.len < min_member_length:
                    diffrence = min_member_length - member.len
                    cost: torch.Tensor = abs(
                        diffrence*torch.norm(current_grad))*2
                    cost.backward()

            # max length is not tested yet
            if max_member_length is not None:
                if member.len > max_member_length:
                    diffrence = member.len - max_member_length
                    cost: torch.Tensor = abs(
                        diffrence*torch.norm(current_grad))*2
                    cost.backward()

    def __optimize_member_forces(self, joint: Joint, max_compresive_force=None, max_tensile_force=None):
        self.solve_supports()
        self.solve_members()
        current_grad = joint.vector.grad.data if joint.vector.grad is not None else 1
        for member in self.__members:
            if max_compresive_force is not None:
                if member.force > max_compresive_force and member.force_type == "c":
                    force = member.force
                    cost: torch.Tensor = abs(
                        force - max_compresive_force)*torch.norm(current_grad)*2
                    cost.backward(retain_graph=True)
                    # get gradient for each joint

            if max_tensile_force is not None:
                if member.force > max_tensile_force and member.force_type == "t":
                    force = member.force
                    cost: torch.Tensor = abs(
                        force - max_tensile_force)*torch.norm(current_grad)*2
                    cost.backward(retain_graph=True)

    def optimize_cost(
            self,
            member_cost,
            joint_cost,
            lr=0.01,
            epochs=10,
            print_cost=True,
            show_at_epoch=True,
            min_member_length=None,
            max_member_length=None,
            max_tensile_force=None,
            max_compresive_force=None,
            progress_bar=True,
    ):
        """Will optimize price."""

        if progress_bar:
            epochs = tqdm(range(epochs))
        else:
            epochs = range(epochs)

        for _ in epochs:

            cost: torch.Tensor = self.get_cost(member_cost, joint_cost)
            cost.backward()

            if print_cost:
                print(cost)
            if show_at_epoch:
                self.show()

            # self.solve_members()
            for joint in self.__joints:
                if joint.track_grad == True:

                    # apend min length gradients to cost gradeints
                    if min_member_length is not None or max_member_length is not None:
                        self.__optimize_member_length(
                            joint, min_member_length, max_member_length)

                    if max_compresive_force is not None or max_tensile_force is not None:
                        self.__optimize_member_forces(
                            joint, max_compresive_force, max_tensile_force)

                    # get gradient for each joint
                    grad = joint.vector.grad.data

                    # move joints
                    new_x = joint.x_coordinate - grad[0]*lr
                    new_y = joint.y_coordinate - grad[1]*lr
                    joint.set_x(new_x)
                    joint.set_y(new_y)

                    # reset all gradeints
                    grad.zero_()

                    # if max_compresive_force is not None or max_tensile_force is not None:
                    #     self.__optimize_member_forces(
                    #         joint, max_compresive_force, max_tensile_force)

            # for joint in self.__joints:
            #     if joint.track_grad == True:
            #         if min_member_length is not None or max_member_length is not None:
            #             self.__optimize_member_length(
            #                 joint, min_member_length, max_member_length)
