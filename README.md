# Trussty

Trusty is a Python library for modeling and analyzing trusses using tensors for computational efficiency. It leverages the power of PyTorch for gradient tracking, enabling force and member optimization, and supports structural analysis with joints, members, forces, and supports. This library provides a flexible and intuitive framework to model 2D truss structures and solve for reactions, internal member forces, and optimize truss designs.

## Features

- **Joint and Member Representation**: Define joints with x, y coordinates, and add members to connect them.
- **Force Application and Calculation**: Apply forces on joints and calculate the resultant internal forces in truss members.
- **Support Types**: Pin, roller, fixed, and custom supports can be added to joints.
- **Structural Analysis**: Solve for reactions at supports and member forces under loading.
- **Optimization Capabilities**: Minimize the cost of the truss structure by optimizing member lengths and forces.
- **PyTorch Integration**: Uses PyTorch to handle vector operations and gradient tracking for optimization.

## Installation

Install with pip:

```bash
pip install trussty
```

Or clone the repository and install dependancies:

```bash
git clone https://github.com/yourusername/truss-analysis.git
cd truss-analysis
pip install -r requirements.txt
```

## Usage

### Create a Mesh

Define a mesh of joints and members:

```python
from trussty import Joint, Member, Mesh

# Create joints
joint1 = Joint(x_coordinate=0, y_coordinate=0)
joint2 = Joint(x_coordinate=1, y_coordinate=0)
joint3 = Joint(x_coordinate=0, y_coordinate=1)

# Create a member
member1 = Member(joint_a=joint1, joint_b=joint2)

# Initialize a mesh
mesh = Mesh()
mesh.add_joint(joint1)
mesh.add_joint(joint2)
mesh.add_member(member1)

# Print mesh members
mesh.print_members()
```

### Apply Forces

You can apply forces to joints:

```python
from trussty import Force

force = Force(joint=joint2, x_component=100.0, y_component=-50.0)
mesh.apply_force(force)
```

### Solve Reactions and Forces

Once the forces and supports are defined, you can solve the system:

```python
mesh.solve_supports(print_reactions=True)
mesh.solve_members()
```

### Optimization

Optimize the cost of the truss structure by minimizing member length and forces:

```python
mesh.optimize_cost(member_cost=10.0, joint_cost=5.0)
```

## CSV Import

You can import node and member data from CSV files:

```python
mesh.from_csv(path_to_node_csv="nodes.csv", path_to_member_csv="members.csv")
```

## Visualize the Truss

You can plot the truss using Matplotlib:

```python
mesh.show(show=True)
```

## Saving and Loading

Save the mesh object for future use:

```python
mesh.save(name="truss_model", relative_path="./models/")
```

## Example

```python
# Example of a simple truss structure with optimization and visualization
from trussty import Joint, Member, Mesh, Support, Force

# Create joints
joint1 = Joint(x_coordinate=0, y_coordinate=0)
joint2 = Joint(x_coordinate=5, y_coordinate=0)
joint3 = Joint(x_coordinate=2.5, y_coordinate=5)

# Create members
member1 = Member(joint_a=joint1, joint_b=joint2)
member2 = Member(joint_a=joint1, joint_b=joint3)
member3 = Member(joint_a=joint2, joint_b=joint3)

# Initialize mesh and add joints/members
mesh = Mesh()
mesh.add_joint(joint1)
mesh.add_joint(joint2)
mesh.add_joint(joint3)
mesh.add_member(member1)
mesh.add_member(member2)
mesh.add_member(member3)

# Add supports
mesh.add_support(Support(joint1, base="p"))
mesh.add_support(Support(joint2, base="r"))

# Apply forces
force = Force(joint3, x_component=0, y_component=-100)
mesh.apply_force(force)

# Solve system and visualize
mesh.solve_supports()
mesh.solve_members()
mesh.show(show=True)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
