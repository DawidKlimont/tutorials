from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.solver import Solver

from fenicsprecice import Adapter
from fenics import SubDomain, near, Point, RectangleMesh, FunctionSpace, VectorFunctionSpace, interpolate, Expression

from numpy import vectorize
from sympy import Symbol, Function

import modulus
from modulus.sym.hydra import ModulusConfig
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE


class StraightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[0], 1.0, tol):
            return True
        else:
            return False
        
class HeatPDE(PDE):
        def __init__(self, alpha, beta):
                x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
                input_variables = {"x": x, "y": y, "t": t}
                u = Function("u")(*input_variables)
                u_x = Function("u_x")(*input_variables)

                self.equations = {}
                self.equations["heat_equation"] = u.diff(t) - (u.diff(x,2)+u.diff(y,2)) - (beta-2-2*alpha)/10
                self.equations["flux_x"] = u.diff(x) - u_x       

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig):

    dt = 0.1
    t_coupling = 0.0
    n = 0
    alpha = 3
    beta = 1.2

    #TODO init model and initial training
    geometry = Rectangle((0,0),(1,1))
    u_net = FullyConnectedArch(
        input_keys = [Key("x"), Key("y"), Key("t")],
        output_keys = [Key("u"), Key("u_x")],
        layer_size = 128,
        nr_layers = 4,
    )

    x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
    equation = HeatPDE(alpha, beta)
    nodes = equation.make_nodes() + [u_net.make_node("u_network")]

    boundary_condition = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geometry,
        outvar = {"u": ((1+x*x+y*y*alpha)/10)},
        batch_size = 1_000,
        parameterization={t: 0.0},
    )

    initial_condition = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geometry,
        outvar = {"u": ((1+x*x+y*y*alpha)/10)},
        batch_size = 1_000,
        parameterization = {t: 0.0}
    )

    domain = Domain()
    domain.add_constraint(initial_condition)
    domain.add_constraint(boundary_condition)
    solver = Solver(cfg=cfg, domain=domain)
    solver.solve()
    


    coupled_boundary_expression = []
    while precice.is_coupling_ongoing():
        if precice.requires_writing_checkpoint():
            precice.store_checkpoint(f_N_function, t_coupling, n)

        read_data = precice.read_data(dt)
        precice.update_coupling_expression(coupling_expression, read_data)
        coupled_boundary_expression.append( (t_coupling+dt, vectorize(coupling_expression)) )


        precice.write_data(f_N_function)#Placeholder later actual function derived from pointvalues in modulusmodel
        precice.advance(dt)
        #TODO write actual model data

        
        if precice.requires_reading_checkpoint():
            #TODO train model
            domain = Domain()
            solver = Solver(cfg=cfg, domain=domain)
            _, t_coupling, n = precice.retrieve_checkpoint()
            coupled_boundary_expression = [] #TODO only remove n_diff last entries -> timeframe capable

        else:
            t_coupling += dt
            n += 1

    precice.finalize()

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 11, 11, diagonal="left")
V = FunctionSpace(mesh, 'P', 2)
W = VectorFunctionSpace(mesh, 'P', 1).sub(0).collapse()
f_N_function = interpolate(Expression("2", degree=0), W)
coupling_boundary = StraightBoundary()

precice = Adapter(adapter_config_filename="precice-adapter-config.json")
precice.initialize(coupling_boundary, read_function_space=V, write_object=f_N_function) #Needs to be done before run() since decorator changes location?
coupling_expression = precice.create_coupling_expression()

print("Starting")
run()
print("Finished")