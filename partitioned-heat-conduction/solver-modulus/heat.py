from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.solver import Solver

from fenicsprecice import Adapter
from fenics import SubDomain, near, Point, RectangleMesh, FunctionSpace, VectorFunctionSpace, interpolate, Expression

import torch
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
    def __init__(self, alpha, beta, scaling):
            x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
            input_variables = {"x": x, "y": y, "t": t}
            u = Function("u")(*input_variables)
            u_x = Function("u_x")(*input_variables)

            self.equations = {}
            self.equations["heat_equation"] = u.diff(t) - (u.diff(x,2)+u.diff(y,2)) - (beta-2-2*alpha)/scaling
            self.equations["flux_x"] = u.diff(x) - u_x 

class Modulus_Helper():
    def __init__(self, model, cfg, alpha, beta, scaling):
        self.alpha = alpha 
        self.beta = beta
        self.scaling = scaling

        self.cfg = cfg
        self.steps_per_iter = self.cfg.training.max_steps
        self.cfg.training.max_steps = 0
        self.total_steps = 0

        self.model = model
        self.geometry = Rectangle((0,0),(1,1))
        self.nodes = HeatPDE(alpha, beta, scaling).make_nodes() + [model.make_node("u_network")]

    def train_model(self, end_time, coupled_boundary_expressions):
        self.cfg.training.max_steps+=self.steps_per_iter
        if self.total_steps>=self.cfg.training.max_steps:
            return
        
        tolerance = 1e-10
        x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
        time_range = {t: (0.0, end_time)}

        initial_condition = PointwiseInteriorConstraint(
            nodes = self.nodes,
            geometry = self.geometry,
            outvar = {"u": ((1+x*x+y*y*self.alpha)/self.scaling)},
            batch_size = 1_000,
            parameterization = {t: 0.0}
        )

        interior_constraint = PointwiseInteriorConstraint(
            nodes = self.nodes,
            geometry = self.geometry,
            outvar = {"heat_equation": 0, "flux_x": 0},
            batch_size = 5_000,
            parameterization=time_range
        )

        boundary_condition = PointwiseBoundaryConstraint(
            nodes = self.nodes,
            geometry = self.geometry,
            outvar = {"u": ((1+x*x+y*y*self.alpha+t*self.beta)/self.scaling)},
            batch_size = 1_000,
            criteria=x<1.0-tolerance,
            parameterization=time_range,
        )

        coupled_constraints = []
        for t_expr, expression in coupled_boundary_expressions:
            coupled_constraints.append(
                PointwiseBoundaryConstraint(
                    nodes = self.nodes,
                    geometry = self.geometry,
                    outvar = {"u": lambda x,y,t: expression(x,y)/self.scaling},
                    batch_size = 10,
                    criteria=x>1.0-tolerance,
                    parameterization={t: t_expr},
                )
            )

        domain = Domain()
        domain.add_constraint(initial_condition)
        domain.add_constraint(interior_constraint)
        domain.add_constraint(boundary_condition)
        for constraint in coupled_constraints:
            domain.add_constraint(constraint)

        solver = Solver(cfg=self.cfg, domain=domain)
        solver.solve()
        self.total_steps = solver.load_step()


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig):

    dt = 0.01
    t_coupling = 0.0
    n = 0
    alpha = 3
    beta = 1.2
    scaling = 10
    coupled_boundary_expressions = []

    #init model and initial training also loads pre trained model if in output folder
    u_net = FullyConnectedArch(
        input_keys = [Key("x"), Key("y"), Key("t")],
        output_keys = [Key("u"), Key("u_x")],
        layer_size = 128,
        nr_layers = 4,
    )
    modulus = Modulus_Helper(u_net, cfg, alpha, beta, scaling)
    modulus.train_model(0.0, coupled_boundary_expressions)
    
    while precice.is_coupling_ongoing():
        if precice.requires_writing_checkpoint():
            precice.store_checkpoint(f_N_function, t_coupling, n)

        read_data = precice.read_data(dt)
        precice.update_coupling_expression(coupling_expression, read_data)
        coupled_boundary_expressions.append( (t_coupling+dt, vectorize(coupling_expression)) )

        

        inputs = precice._owned_vertices.get_coordinates()
        input_dict = {
           "x": torch.tensor([x for x,_ in inputs], dtype=torch.float32, device="cuda").unsqueeze(-1),
           "y": torch.tensor([y for _,y in inputs], dtype=torch.float32, device="cuda").unsqueeze(-1),
           "t": torch.tensor([t_coupling+dt for _ in inputs], dtype=torch.float32, device="cuda").unsqueeze(-1)
        }
        u_net.eval()
        output = u_net(input_dict)["u_x"].squeeze().detach().cpu().numpy()*scaling
        u_net.train()

        precice._participant.write_data(
            precice._config.get_coupling_mesh_name(),
            precice._config.get_write_data_name(),
            precice._precice_vertex_ids,
            output
        )
        precice.advance(dt)

        
        if precice.requires_reading_checkpoint():
            modulus.train_model(t_coupling+dt, coupled_boundary_expressions)
            _, t_coupling, n = precice.retrieve_checkpoint()
            coupled_boundary_expressions = coupled_boundary_expressions[:n]

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


# 1 timeframe + 10 iters + 100(very low total -> 1000) training steps per iter -> 5 mins of training
# 1 timeframe + 100 iters ?->? 50 mins
# 10 timeframes + 100 iters ?->? 500 mins <- 8 hours could do over night and then store on github