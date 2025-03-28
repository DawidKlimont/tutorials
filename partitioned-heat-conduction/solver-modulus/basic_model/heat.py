from sympy import Symbol, Function

from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.activation import Activation

from modulus.sym.key import Key
from modulus.sym.domain import Domain
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from modulus.sym.solver import Solver
from modulus.sym.utils.io import ValidatorPlotter
from modulus.sym.eq.pde import PDE

from modulus.sym.hydra import ModulusConfig
import modulus
import torch

class HeatEquation2D(PDE):
    def __init__(self, alpha, beta, scaling):
        x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
        input_variables = {"x": x, "y": y, "t": t}
        u = Function("u")(*input_variables)
        self.equations = {}
        self.equations["heat_equation"] = u.diff(t) -(u.diff(x,2)+u.diff(y,2)) -(beta-2-2*alpha)/scaling

class CustomPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        invar_subset = {"x": invar["x"],"y": invar["y"]}
        return super().__call__(invar_subset, true_outvar, pred_outvar)
    
def initialize_neural_network():
	#neural network initialize
	u_net = FullyConnectedArch(
		input_keys = [Key("x"), Key("y"), Key("t")],
		output_keys = [Key("u")],		
		activation_fn = Activation.TANH,	
		layer_size = 128,
		nr_layers = 7,
	)
	return u_net

def initialize_nodes_and_geometry(u_net, alpha, beta, scaling):
	geometry = Rectangle((0,0),(1,1))
	eq = HeatEquation2D(alpha, beta, scaling)
	nodes = eq.make_nodes() + [u_net.make_node("u_network")]
	return nodes, geometry

def initialize_constraints(nodes, geometry, alpha, beta, scaling):
	constraints = []
	x,y,t = Symbol("x"), Symbol("y"), Symbol("t")
	time_range = {t: (0.0,1.0)}

	initial_condition = PointwiseInteriorConstraint(
		nodes = nodes,
		geometry = geometry,
		outvar = {"u": (1 + x*x + alpha*y*y)/(scaling)},
		batch_size = 1_000,
		parameterization = {t: 0.0},
		fixed_dataset  = False
	)

	boundary_condition = PointwiseBoundaryConstraint(
		nodes = nodes,
		geometry = geometry,
		outvar = {"u": (1 + x*x + alpha*y*y + beta*t)/(scaling)},
		batch_size = 1_000,	
		parameterization=time_range,
		fixed_dataset  = False
	)
      
	interior_constraint = PointwiseInteriorConstraint(
		nodes = nodes,
		geometry = geometry,
		outvar = {"heat_equation": 0},
		batch_size = 10_000,
		parameterization=time_range,
		fixed_dataset  = False
	)

	constraints.append(initial_condition)
	constraints.append(boundary_condition)
	constraints.append(interior_constraint)
	return constraints

def initialize_validator(nodes, alpha, beta, scaling):
	validators = []
	c, t = 10, 1.0 
	X, Y = torch.meshgrid(torch.linspace(0, 1, c), torch.linspace(0, 1, c), indexing="ij")
	invar = {"x": X.reshape(-1, 1), "y": Y.reshape(-1, 1), "t": torch.ones(c*c, 1)*t}
	outvar = {"u": (1+invar["x"]*invar["x"]+alpha*invar["y"]*invar["y"]+beta*invar["t"])/scaling}
	validator = PointwiseValidator(
		invar = invar,
		true_outvar = outvar,
		nodes = nodes,
		batch_size=c*c,
		plotter=CustomPlotter(),
	)
	validators.append(validator)
	return validators

def initialize_domain(constraints, validators):
	domain = Domain()		
	for constraint in constraints:
		domain.add_constraint(constraint)
	for validator in validators:
		domain.add_validator(validator)
	return domain

def train_model(domain, cfg):
	solver = Solver(cfg=cfg, domain=domain)
	solver.solve()

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig):
     
	alpha = 3.0
	beta = 1.2
	scaling = 10.0

	u_net = initialize_neural_network()
	nodes, geometry = initialize_nodes_and_geometry(u_net, alpha, beta, scaling)
	constraints = initialize_constraints(nodes, geometry, alpha, beta, scaling)
	validators = initialize_validator(nodes, alpha, beta, scaling)
	domain = initialize_domain(constraints, validators)
	train_model(domain, cfg)
		
run()
