import torch

def sigreg(
    x: torch.Tensor, 
    global_step: int, 
    num_slices: int = 256,
    num_points: int = 17,
) -> torch.Tensor:
    """
    SIGReg: Sliced Isotropic Gaussian Regularization.
    
    Projects embeddings onto random 1D directions and checks if 
    each projection is Gaussian using Epps-Pulley characteristic function test.
    
    Args:
        x: [N, K] tensor of embeddings (flattened batch*seq, hidden)
        global_step: for synchronized random slice sampling
        num_slices: number of random projection directions
        num_points: integration points for characteristic function
    
    Returns:
        Scalar loss measuring deviation from isotropic Gaussian
    """
    dev = dict(device=x.device)
    
    # Synchronized random projections
    g = torch.Generator(**dev)
    g.manual_seed(global_step)
    
    proj_shape = (x.size(1), num_slices)  # [hidden, num_slices]
    A = torch.randn(proj_shape, generator=g, **dev)
    A = A / A.norm(p=2, dim=0, keepdim=True)  # Normalize columns
    
    # Project x onto random directions: [N, num_slices]
    x_proj = x @ A
    
    # Standardize each slice (so we compare to N(0,1))
    x_proj = (x_proj - x_proj.mean(dim=0, keepdim=True)) / (x_proj.std(dim=0, keepdim=True) + 1e-8)
    
    # Epps-Pulley test using characteristic functions
    t = torch.linspace(-5, 5, num_points, **dev)  # Integration points
    
    # Theoretical CF for N(0,1): exp(-t²/2)
    exp_f = torch.exp(-0.5 * t ** 2)
    
    # Empirical CF: E[exp(i*t*x)] for each slice
    x_t = x_proj.unsqueeze(2) * t  # [N, num_slices, num_points]
    ecf = (1j * x_t).exp().mean(dim=0)  # [num_slices, num_points]
    
    # Weighted L2 distance between empirical and theoretical CF
    # Weight by exp_f to focus on important region
    err = (ecf - exp_f).abs().square() * exp_f
    
    # Integrate over t, average over slices
    loss = torch.trapezoid(err, t, dim=1).mean()
    
    return loss

def compute_sigreg_loss(z_H: torch.Tensor, z_L: torch.Tensor, global_step: int, num_slices: int = 256) -> torch.Tensor:
    """Apply SIGReg to hidden states."""
    # Flatten to [batch*seq, hidden]
    z_H_flat = z_H.reshape(-1, z_H.shape[-1])
    z_L_flat = z_L.reshape(-1, z_L.shape[-1])
    
    loss_H = sigreg(x=z_H_flat, global_step=global_step, num_slices=num_slices)
    loss_L = sigreg(x=z_L_flat, global_step=global_step, num_slices=num_slices)
    
    return loss_H + loss_L