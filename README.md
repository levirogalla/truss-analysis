# Truss Analysis Package

This package gives functionality to analyse trusses.

## How to use

Create a mesh using Mesh class.

Add joints, members, forces, and supports using coresponding data classes.

Changes to any of the above classes are done in the same memory location. Meaning changing a joint outside of a mesh will also change the joint inside the mesh.

Supports skyciv node and member cvs files to create mesh from cvs.
