from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.solver import Solver

from fenicsprecice import Adapter
from fenicsprecice.adapter_core import convert_fenics_to_precice
from fenics import SubDomain, near, Point, RectangleMesh, FunctionSpace, VectorFunctionSpace, interpolate, Expression

import torch
from numpy import vectorize
from sympy import Symbol, Function

import modulus
from modulus.sym.hydra import ModulusConfig
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.activation import Activation
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE

from modulus.sym.utils.io import ValidatorPlotter
from modulus.sym.domain.validator import PointwiseValidator

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
            self.equations = {}
            self.equations["heat_equation"] = u.diff(t) - (u.diff(x,2)+u.diff(y,2)) - (beta-2-2*alpha)/scaling

class CustomPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        pred_outvar["u"] = pred_outvar["u"]*10
        true_outvar["u"] = true_outvar["u"]*10
        invar_subset = {"x": invar["x"], "y": invar["y"]}
        return super().__call__(invar_subset, true_outvar, pred_outvar)

class ModulusHelper():
    def __init__(self, model, cfg, alpha, beta, scaling):
        self.alpha, self.beta, self.scaling = alpha, beta, scaling

        self.cfg = cfg
        self.steps_per_iter = self.cfg.training.max_steps
        self.cfg.training.max_steps = 0
        self.total_steps = 0

        self.model = model
        self.geometry = Rectangle((0,0),(1,1))
        self.nodes = HeatPDE(alpha, beta, scaling).make_nodes() + [model.make_node("u_network")]
        self.domain = Domain()
        self.solver = Solver(cfg=self.cfg, domain=self.domain)

    def train_model(self, end_time, coupled_boundary_expressions,  initial=False):
        if self.total_steps>=self.cfg.training.max_steps:
            return
        
        x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
        time_range = {t: (0.0, end_time)}

        initial_condition = PointwiseInteriorConstraint(
            nodes = self.nodes,
            geometry = self.geometry,
            outvar = {"u": ((1+x*x+y*y*self.alpha)/10)},
            batch_size = 1_000,
            parameterization = {t: 0.0}
        )

        interior_constraint = PointwiseInteriorConstraint(
            nodes = self.nodes,
            geometry = self.geometry,
            outvar = {"heat_equation": 0},
            batch_size = 10_000,
            parameterization=time_range
        )

        boundary_condition = PointwiseBoundaryConstraint(
            nodes = self.nodes,
            geometry = self.geometry,
            outvar = {"u": ((1+x*x+y*y*self.alpha+t*self.beta)/10)},
            batch_size = 3_000,
            criteria=x<1.0,
            parameterization=time_range,
        )

        coupled_constraints = []
        for t_expr, expression in coupled_boundary_expressions:
            coupled_constraints.append(
                PointwiseBoundaryConstraint(
                    nodes = self.nodes,
                    geometry = self.geometry,
                    outvar = {"u": lambda x, y, t: expression(x,y)/10},
                    batch_size = 40,
                    criteria=(x>0.5) & (y<1.0) & (y>0.0),
                    parameterization={t: t_expr},
                )
            )

        
        self.domain.constraints.clear()
        self.domain.validators.clear()
        self.domain.add_constraint(initial_condition)
        if not initial:
            self.domain.add_constraint(interior_constraint)
            self.domain.add_constraint(boundary_condition)
            for constraint in coupled_constraints:
                self.domain.add_constraint(constraint)


        c = 1000
        x_vals = torch.linspace(0, 1, c)
        y_vals = torch.linspace(0, 1, c)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
        invar = {
            "x": X,
            "y": Y,
            "t": torch.ones(c*c, 1)
        }
        outvar = {
            "u": (1+invar["x"]*invar["x"]+self.alpha*invar["y"]*invar["y"]+self.beta*invar["t"])/self.scaling,
        }
        validator = PointwiseValidator(
            invar = invar,
            true_outvar = outvar,
            nodes = self.nodes,
            batch_size=c*c,
            plotter=CustomPlotter(),
        )
        self.domain.add_validator(validator)


        self.solver.max_steps+=self.steps_per_iter
        self.solver.solve()
        self.total_steps = self.solver.load_step()

def read(dt):
    read_data = precice.read_data(dt)
    precice.update_coupling_expression(coupling_expression, read_data)

def write(u_net, dt, t_coupling):
    inputs = precice._owned_vertices.get_coordinates()
    input_dict = {
        "x": torch.tensor([[x] for x,_ in inputs], dtype=torch.float32, device="cuda", requires_grad=True),
        "y": torch.tensor([[y] for _,y in inputs], dtype=torch.float32, device="cuda", requires_grad=True),
        "t": torch.tensor([[t_coupling+dt] for _ in inputs], dtype=torch.float32, device="cuda", requires_grad=True) 
    }
    u_net.eval()
    res = u_net(input_dict)["u"]
    res.backward(torch.tensor([[1.] for _ in res], dtype=torch.float32, device="cuda"))
    output = input_dict["x"].grad.squeeze().detach().cpu().numpy()*10
    u_net.train()
    
    precice._participant.write_data(
        precice._config.get_coupling_mesh_name(),
        precice._config.get_write_data_name(),
        precice._precice_vertex_ids,
        output
    )
    precice.advance(dt)

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig):

    dt, t_coupling, n = 0.01, 0.0, 0
    alpha, beta, scaling = 3.0, 1.2, 10.0

    coupled_boundary_expressions = []
    u_net = FullyConnectedArch(
        input_keys = [Key("x"), Key("y"), Key("t")],
        output_keys = [Key("u")],
        activation_fn = Activation.TANH,	
        layer_size = 128,
        nr_layers = 7,
    )
    modulus = ModulusHelper(u_net, cfg, alpha, beta, scaling)
    modulus.train_model(0.0, coupled_boundary_expressions, initial=True)

    while precice.is_coupling_ongoing():
        if precice.requires_writing_checkpoint():
            precice.store_checkpoint(dummy, t_coupling, n)

        read(dt)
        coupled_boundary_expressions.append( (t_coupling+dt, vectorize(coupling_expression)) )
        write(u_net, dt, t_coupling)

        if precice.requires_reading_checkpoint():
            modulus.train_model(t_coupling+dt, coupled_boundary_expressions)
            _, t_coupling, n = precice.retrieve_checkpoint()
            coupled_boundary_expressions = coupled_boundary_expressions[:n] # remove till n entries in expressions
        else:
            t_coupling += dt
            n += 1

    precice.finalize()

def initialize_fenics_and_adapter():
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), 11, 11, diagonal="left")
    V = FunctionSpace(mesh, 'P', 2)
    W = VectorFunctionSpace(mesh, 'P', 1).sub(0).collapse()
    f_N_function = interpolate(Expression("2", degree=0), W)
    coupling_boundary = StraightBoundary()
    precice = Adapter(adapter_config_filename="precice-adapter-config.json")
    precice.initialize(coupling_boundary, read_function_space=V, write_object=f_N_function) #Needs to be done before run() since decorator changes location?
    coupling_expression = precice.create_coupling_expression()
    dummy = interpolate(Expression("0", degree=0), W)#used for the checkpoints, doesnt actually does anything
    return precice, coupling_expression, dummy
    # Create mesh and function space

precice, coupling_expression, dummy = initialize_fenics_and_adapter()
run()

