"""Formal v5 compression math implementation."""
import math

def compress_with_exceptions(partition, *, distortion_weight=1., tolerance_sd=.25):
    """Exact finite-codebook rate-distortion optimizer for leaf means.

    Cost = default identifier and explicitly named exception leaves.
    No fixed cost is charged for numerical values; this is a proxy objective,
    not a literal bit encoding of the numerical evidence. Conditional
    on a default, exception decisions are independent. Distortion is weighted
    squared error in root-SD units. All errors > tolerance_sd or crossing the
    root comparison sign MUST be exceptions, regardless of coding cost.
    Scope, support and actual within-group SD are never approximated.
    """
    leaves = partition["leaves"]
    root = partition["total"]
    scale = partition["scale"]
    bases = sorted(set([root["mean"], *[l["mean"] for l in leaves]]))
    name_bits = math.log2(max(2, len(leaves)))
    exception_cost = name_bits
    best = None
    for base in bases:
        exceptions, defaults = [], []
        loss = 0.
        for i, leaf in enumerate(leaves):
            error = (leaf["mean"] - base)**2 / scale
            distortion = leaf["n"] * error
            opposite = ((leaf["mean"] - root["mean"]) * (base - root["mean"]) < -1e-12)
            mandatory = error > tolerance_sd**2 + 1e-12 or opposite
            if mandatory or distortion_weight * distortion > exception_cost:
                exceptions.append(i)
            else:
                defaults.append(i)
                loss += distortion
        bits = name_bits + len(exceptions) * exception_cost
        score = bits + distortion_weight * loss
        candidate = {"default_mean": base, "exception_indices": exceptions,
                     "default_indices": defaults, "distortion": loss,
                     "code_length": bits, "objective": score}
        if best is None or score < best["objective"] - 1e-10:
            best = candidate
    assert best is not None
    reconstruction = [l["mean"] if i in best["exception_indices"] else best["default_mean"]
                      for i, l in enumerate(leaves)]
    maximum_error = max(abs(a - l["mean"]) / math.sqrt(scale)
                        for a, l in zip(reconstruction, leaves))
    if maximum_error > tolerance_sd + 1e-8:
        raise ArithmeticError("compression error exceeds hard bound")
    return {**best, "reconstructed_means": reconstruction,
            "max_error_in_root_sd": maximum_error, "tolerance_sd": tolerance_sd,
            "distortion_weight": distortion_weight,
            "guarantee_scope": "structured conditional means only; binding and free prose not certified"}

