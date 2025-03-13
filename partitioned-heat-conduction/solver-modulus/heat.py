from fenicsprecice import Adapter
from fenics import SubDomain, near, Point, RectangleMesh, FunctionSpace, VectorFunctionSpace, interpolate, Expression

class StraightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[0], 1.0, tol):
            return True
        else:
            return False

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 11, 11, diagonal="left")
V = FunctionSpace(mesh, 'P', 2)
W = VectorFunctionSpace(mesh, 'P', 1).sub(0).collapse()
f_N_function = interpolate(Expression("2", degree=0), W)
coupling_boundary = StraightBoundary()

precice = Adapter(adapter_config_filename="precice-adapter-config.json")
precice.initialize(coupling_boundary, read_function_space=V, write_object=f_N_function)
coupling_expression = precice.create_coupling_expression()

dt = 0.1
t = 0.0
n = 0

#TODO init model and initial training

while precice.is_coupling_ongoing():
    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(f_N_function, t, n)

    read_data = precice.read_data(dt)
    precice.update_coupling_expression(coupling_expression, read_data)
    #TODO store expressions

    precice.write_data(f_N_function)#Placeholder later actual function derived from pointvalues in modulusmodel
    precice.advance(dt)
    #TODO write actual model data

    
    if precice.requires_reading_checkpoint():
        #TODO train model
        _, t, n = precice.retrieve_checkpoint()

    else:
        t += dt
        n += 1

precice.finalize()
print("Finished")