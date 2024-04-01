import torch
import torch.nn.functional as F
import itertools

from scipy.special import binom, comb
from math import floor

import torch.nn as nn

# note this does not do the constant factor, that is left up to the user!
def integrate(poly_mat, axis_name="x"):
    orig_names = poly_mat.names
    new_names = list(filter(lambda x: x != axis_name, poly_mat.names)) + [axis_name]

    int_mat = poly_mat.align_to(*new_names)

    # technically this is said constant fact, we could refactor this out as an arg.
    int_mat = F.pad(input=int_mat.rename(None), pad=(1, 0), mode='constant', value=0)

    divisors = torch.arange(int_mat.shape[-1])
    divisors[0] = 1.
    new_poly = torch.einsum("...y,y->...y", int_mat, 1 / divisors).refine_names(*new_names)

    return new_poly.align_to(*orig_names)

def eval_poly(poly_mat, val, axis_name="x"):
    orig_names = poly_mat.names
    other_axes = list(filter(lambda x: x != axis_name, poly_mat.names))
    new_names = [axis_name] + other_axes
    axis_deg = poly_mat.shape[poly_mat.names.index(axis_name)]

    eval_mat = poly_mat.align_to(*new_names)
    res_poly = torch.einsum("x...,x->...", eval_mat.rename(None), (val**torch.arange(axis_deg)).float()).refine_names(*other_axes)
    return res_poly

def defn_integral(poly_mat, bounds, axis_name="x"):
    int_poly_mat = integrate(poly_mat, axis_name)
    return eval_poly(int_poly_mat, bounds[1], axis_name) - eval_poly(int_poly_mat, bounds[0], axis_name)

# really inneffiecient torch code.
def poly_mul(poly_mat_1, poly_mat_2):
    orig_names = poly_mat_1.names
    poly_mat_2 = poly_mat_2.align_to(*orig_names)
    
    poly_mat_3 = torch.zeros((torch.tensor(poly_mat_1.shape) -1 + torch.tensor(poly_mat_2.shape) -1 +1).tolist())
    for p1_pow in itertools.product(*[range(shape) for shape in poly_mat_1.shape]):
        for p2_pow in itertools.product(*[range(shape) for shape in poly_mat_2.shape]):
            new_pos = tuple(x+y for x,y in zip(p1_pow, p2_pow))
            poly_mat_3[new_pos] += poly_mat_1[p1_pow] * poly_mat_2[p2_pow]
    
    return poly_mat_3.refine_names(*orig_names)

def poly_mul_1d(poly_mat_1, poly_mat_2):
    assert poly_mat_1.names == poly_mat_2.names
    assert len(poly_mat_1.names) == 1
    
    kernel = poly_mat_2.rename(None).flip(0)
    kernel = F.pad(kernel, (0, kernel.shape[0] -1), mode="constant", value=0.)
    p2_deg = poly_mat_2.shape[0]
    
    padded_poly_mat_1 = F.pad(poly_mat_1.rename(None), (0, p2_deg))
    
    new_poly_mat = F.conv1d(padded_poly_mat_1.view(1, 1, -1), kernel.view(1, 1, -1), stride=1, padding="same", bias=None).view(-1)
    
    return new_poly_mat.refine_names(*poly_mat_1.names)

def print_poly(poly, print_zeros = False, print_one_coefs = False, print_zero_deg = False, print_one_deg_no_exp = True):
    axis_iterators = [range(shape) for shape in poly.shape]
    str_rep = ""
    for axis_pows in itertools.product(*axis_iterators):
        coef = poly[tuple(axis_pows)]
        if coef == 0 and not print_zeros: continue
        if coef < 0:
            str_rep = str_rep[:-1]
        if coef == 1 and print_one_coefs:
            str_rep += f"{coef}*"
        else:
            str_rep += f"{coef}*"
        
        for axis_pow, axis_name in zip(axis_pows, poly.names):
            if axis_pow == 0 and not print_zero_deg: continue
            if axis_pow == 1 and print_one_deg_no_exp:
                str_rep += f"{axis_name}"
            else:
                str_rep += f"{axis_name}^{axis_pow}"
        
        if str_rep[-1] == "*": str_rep = str_rep[:-1]
        str_rep += " +"
    str_rep = str_rep[:-2]
    if len(str_rep) == 0: print("0")
    else: print(str_rep)

def expand_poly(poly_mat, axes):
    return poly_mat.align_to(*axes)

def poly1d(coeffs, axis_name="x", axes=None):
    poly_mat = torch.tensor(coeffs).refine_names(axis_name)
    if axes is not None:
        poly_mat = expand_poly(poly_mat, axes)
    return poly_mat

def zero_poly(degs, spec_axes, all_axes=None):
    poly_mat = torch.zeros([deg+1 for deg in degs]).refine_names(*spec_axes)
    if all_axes is None: return poly_mat
    else: return expand_poly(poly_mat, all_axes)

def randn_poly(degs, spec_axes, all_axes=None, scale=1.0):
    poly_mat = torch.randn([deg+1 for deg in degs]).refine_names(*spec_axes)
    poly_mat *= scale
    
    if all_axes is not None:
        poly_mat = expand_poly(poly_mat, all_axes)
    
    return poly_mat

# really expensive
def legendre_coefs(deg):
    poly_mat = torch.zeros((deg+1, deg+1))
    
    for n in range(deg+1):
        for k in range(n+1):
            poly_mat[n, k] = 2**n * binom(n, k) * binom((n+k-1)/2, n)
        
    return poly_mat

def legendre1d_to_pbasis(legendre_poly1d):
    degree = legendre_poly1d.shape[0] -1
    return (legendre_poly1d @ legendre_coefs(degree)).refine_names(*legendre_poly1d.names)

# really expensive
def legendrend_to_pbasis(legendre_polynd):
    original_ordering = legendre_polynd.names
    
    new_poly = legendre_polynd
    for axis in legendre_polynd.names:
        new_axis_order = [axis] + [a for a in legendre_polynd.names if a != axis]
        new_poly = new_poly.align_to(*new_axis_order)
        new_poly = torch.einsum("p...,px->x...", new_poly.rename(None), legendre_coefs(new_poly.shape[0] -1)).refine_names(*new_poly.names)
    
    return new_poly.align_to(*original_ordering)

class CTNHelper:
    def __init__(self) -> None:
        self.all_params = {}
        self.all_axes = None
        self.all_axes_with_batch = None
        
    
    def add(self, name, degs, axes, scale=0.5):
        self.all_params[name] = randn_poly(degs, axes, scale=scale)
    
    def to_dict(self):
        all_axes = set()
        for tensor in self.all_params.values():
            all_axes.update(tensor.names)
        
        all_axes = list(all_axes)
        print(f"{all_axes=}")
        self.all_axes = all_axes
        self.all_axes_with_batch = ["batch"] + all_axes
        
        all_items = list(self.all_params.items())
        for name, tensor in all_items:
            # print(f"{name=}")
            self.all_params[name] = tensor.align_to(*all_axes)
        
        return self.all_params

def toparam(tuple):
    return tuple[0], nn.Parameter(tuple[1])

def batch_eval_poly(poly_mat, val, axis_name="x", auto_batch=True):
    orig_names = poly_mat.names
    other_axes = list(filter(lambda x: x != axis_name and x != "batch", poly_mat.names))
    
    # TODO: check this
    if "batch" not in orig_names and auto_batch:
        new_names = ["batch", axis_name] + other_axes
    elif "batch" in orig_names:
        new_names = ["batch", axis_name] + other_axes
    else:
        new_names = [axis_name] + other_axes
        
    axis_deg = poly_mat.shape[poly_mat.names.index(axis_name)]

    eval_mat = poly_mat.align_to(*new_names)
    res_poly = torch.einsum("ix...,ix->i...", eval_mat.rename(None), (val**torch.arange(axis_deg)).float()).refine_names("batch", *other_axes)
    return res_poly

# really (really) inneffiecient torch code.
def batch_poly_mul(poly_mat_1, poly_mat_2):
    p1_batched = False
    p2_batched = False
    
    nbatches = 1
    p1_shape = None
    p2_shape = None
    
    if "batch" in poly_mat_1.names:
        assert poly_mat_1.names[0] == "batch"
        p1_batched = True
        nbatches = poly_mat_1.shape[0]
        p1_shape = torch.tensor(poly_mat_1.shape[1:])
    else:
        p1_shape = torch.tensor(poly_mat_1.shape)

    if "batch" in poly_mat_2.names:
        assert poly_mat_2.names[0] == "batch"
        p2_batched = True
        nbatches = poly_mat_2.shape[0]
        p2_shape = torch.tensor(poly_mat_2.shape[1:])
    else:
        p2_shape = torch.tensor(poly_mat_2.shape)
    
    res_poly = torch.zeros([nbatches] + (p1_shape -1 + p2_shape -1 +1).tolist())
    
    if p1_batched and p2_batched:
        assert poly_mat_1.shape[0] == poly_mat_2.shape[0]
        for idx, (p1, p2) in enumerate(zip(poly_mat_1, poly_mat_2)):
            bpoly = poly_mul(p1, p2)
            
            if res_poly.names[0] is None:
                res_poly = res_poly.refine_names("batch", *bpoly.names)
                
            res_poly[idx] = bpoly
    elif p1_batched and not p2_batched:
        p2 = poly_mat_2
        for idx, p1 in enumerate(poly_mat_1):
            bpoly = poly_mul(p1, p2)
            
            if res_poly.names[0] is None:
                res_poly = res_poly.refine_names("batch", *bpoly.names)
                
            res_poly[idx] = bpoly
    elif p2_batched and not p1_batched:
        p1 = poly_mat_1
        for idx, p2 in enumerate(poly_mat_2):
            bpoly = poly_mul(p1, p2)
            
            if res_poly.names[0] is None:
                res_poly = res_poly.refine_names("batch", *bpoly.names)
                
            res_poly[idx] = bpoly
    elif not p2_batched and not p1_batched:
        return poly_mul(poly_mat_1, poly_mat_2)
    return res_poly

def batch_integrate(poly_mat, axis_name="x"):
    orig_names = poly_mat.names
    assert orig_names[0] == "batch"
    
    new_names = list(filter(lambda x: x != axis_name and x != "batch", poly_mat.names)) + [axis_name]

    int_mat = poly_mat.align_to("batch", *new_names)

    # technically this is said constant fact, we could refactor this out as an arg.
    int_mat = F.pad(input=int_mat.rename(None), pad=(1, 0), mode='constant', value=0)

    divisors = torch.arange(int_mat.shape[-1])
    divisors[0] = 1.
    new_poly = torch.einsum("i...y,y->i...y", int_mat, 1 / divisors).refine_names("batch", *new_names)

    return new_poly.align_to(*orig_names)

def print_batch_poly(batch_poly_mat):
    assert batch_poly_mat.names[0] == "batch"
    for idx, poly_mat in enumerate(batch_poly_mat):
        print(f"poly[{idx}]: ", end="")
        print_poly(poly_mat)
        
def batch_defn_integral(poly_mat, bounds, axis_name="x"):
    int_poly_mat = batch_integrate(poly_mat, axis_name)
    return batch_eval_poly(int_poly_mat, torch.tensor(bounds[1]).view(-1, 1), axis_name) - batch_eval_poly(int_poly_mat, torch.tensor(bounds[0]).view(-1, 1), axis_name)

def drop_axes(tensor, axes):
    keep_axes = []
    for axis in tensor.names:
        if axis not in axes:
            keep_axes.append(axis)
    
    tensor = tensor.align_to(*keep_axes, *axes)
    
    for _ in axes:
        tensor = tensor.squeeze(-1)
    
    return tensor

def keep_axes(tensor, axes):
    return drop_axes(tensor, list(set(tensor.names).difference(axes)))