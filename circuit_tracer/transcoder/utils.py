
import torch
from itertools import zip_longest
from types import SimpleNamespace

import numpy as np
import torch
import tqdm

def geometric_median_list_of_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``torch.Tensor``.
        Each inner list has the same "shape".
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``torch.Tensor`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        median = weighted_average(points, weights)
        new_weights = weights
        objective_value = geometric_median_objective(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        for _ in range(maxiter):
            prev_obj_value = objective_value
            denom = torch.stack([l2distance(p, median) for p in points])
            new_weights = weights / torch.clamp(denom, min=eps) 
            median = weighted_average(points, new_weights)

            objective_value = geometric_median_objective(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break
        
    median = weighted_average(points, new_weights)  # for autodiff

    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def weighted_average_component(points, weights):
    ret = points[0] * weights[0]
    for i in range(1, len(points)):
        ret += points[i] * weights[i]
    return ret

def weighted_average(points, weights):
    weights = weights / weights.sum()
    return [weighted_average_component(component, weights=weights) for component in zip(*points)]

@torch.no_grad()
def geometric_median_objective(median, points, weights):
    return np.average([l2distance(p, median).item() for p in points], weights=weights.cpu())

@torch.no_grad()
def l2distance(p1, p2):
    return torch.linalg.norm(torch.stack([torch.linalg.norm(x1 - x2) for (x1, x2) in zip(p1, p2)]))


def check_list_of_array_format(points):
	check_shapes_compatibility(points, -1)

def check_list_of_list_of_array_format(points):
	# each element of `points` is a list of arrays of compatible shapes
	components = zip_longest(*points, fillvalue=torch.Tensor())
	for i, component in enumerate(components):
		check_shapes_compatibility(component, i)

def check_shapes_compatibility(list_of_arrays, i):
	arr0 = list_of_arrays[0]
	if not isinstance(arr0, torch.Tensor):
		raise ValueError(
			"Expected points of format list of `torch.Tensor`s.", 
			f"Got {type(arr0)} for component {i} of point 0."
		)
	shape = arr0.shape
	for j, arr in enumerate(list_of_arrays[1:]):
		if not isinstance(arr, torch.Tensor):
			raise ValueError(
				f"Expected points of format list of `torch.Tensor`s. Got {type(arr)}",
				f"for component {i} of point {j+1}."
			)
		if arr.shape != shape:
			raise ValueError(
				f"Expected shape {shape} for component {i} of point {j+1}.",
				f"Got shape {arr.shape} instead."
			)
		

def compute_geometric_median(
	points, weights=None, per_component=False, skip_typechecks=False,
	eps=1e-6, maxiter=100, ftol=1e-20
):
	""" Compute the geometric median of points `points` with weights given by `weights`. 
	"""
	if type(points) == torch.Tensor:
		# `points` are given as an array of shape (n, d)
		points = [p for p in points]  # translate to list of arrays format
	if type(points) not in [list, tuple]:
		raise ValueError(
			f"We expect `points` as a list of arrays or a list of tuples of arrays. Got {type(points)}"
		)
	if type(points[0]) == torch.Tensor: # `points` are given in list of arrays format
		if not skip_typechecks:
			check_list_of_array_format(points)
		if weights is None:
			weights = torch.ones(len(points), device=points[0].device)
		to_return = geometric_median_array(points, weights, eps, maxiter, ftol)
	elif type(points[0]) in [list, tuple]: # `points` are in list of list of arrays format
		if not skip_typechecks:
			check_list_of_list_of_array_format(points)
		if weights is None:
			weights = torch.ones(len(points), device=points[0][0].device)
		if per_component:
			to_return = geometric_median_per_component(points, weights, eps, maxiter, ftol)
		else:
			to_return = geometric_median_list_of_array(points, weights, eps, maxiter, ftol)
	else:
		raise ValueError(f"Unexpected format {type(points[0])} for list of list format.")
	return to_return
		
def geometric_median_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:`n`, whose elements are each a ``torch.Tensor`` of shape ``(d,)``
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value
            norms = torch.stack([torch.linalg.norm((p - median).view(-1)) for p in points])
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)

            objective_value = geometric_median_objective(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break
            
            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def geometric_median_per_component(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``numpy.ndarray``.
        Each inner list has the same "shape".
    :param weights: ``numpy.ndarray`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``numpy.ndarray`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated, one for each component. 
        - `logs`: function values encountered through the course of the algorithm.
    """
    components = list(zip(*points))
    median = []
    termination = []
    logs = []
    new_weights = []
    pbar = tqdm.tqdm(components)
    for component in pbar:
        ret = geometric_median_array(component, weights, eps, maxiter, ftol)
        median.append(ret.median)
        new_weights.append(ret.new_weights)
        termination.append(ret.termination)
        logs.append(ret.logs)
    return SimpleNamespace(median=median, termination=termination, logs=logs)

def weighted_average(points, weights):
    weights = weights / weights.sum()
    ret = points[0] * weights[0]
    for i in range(1, len(points)):
        ret += points[i] * weights[i]
    return ret

@torch.no_grad()
def geometric_median_objective(median, points, weights):
    return np.average([torch.linalg.norm((p - median).reshape(-1)).item() for p in points], weights=weights.cpu())
