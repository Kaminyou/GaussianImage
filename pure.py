import torch


def get_tile_bbox(pix_center, pix_radius, tile_bounds, BLOCK_X=16, BLOCK_Y=16):
    tile_size = torch.tensor(
        [BLOCK_X, BLOCK_Y], dtype=torch.float32, device=pix_center.device
    )
    tile_center = pix_center / tile_size
    tile_radius = pix_radius[..., None] / tile_size

    top_left = (tile_center - tile_radius).to(torch.int32)
    bottom_right = (tile_center + tile_radius).to(torch.int32) + 1
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    return tile_min, tile_max

def compute_cov2d_bounds(l11: torch.Tensor, l12: torch.Tensor, l22: torch.Tensor, eps=1e-6):
    det = l11 * l22 - l12 ** 2
    det = torch.clamp(det, min=eps)  # redundant?
    conic = torch.stack(
        [
            l22 / det,
            -l12 / det,
            l11 / det,
        ],
        dim=-1,
    )  # (..., 3)
    b = (l11 + l22) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    return conic, radius, det > eps


def project_gaussians_2d_torch(
    xyz, # [N, 2]
    cholesky_elements, # [N, 3]
    H,
    W,
    tile_bounds,
):
    N = xyz.shape[0]
    wh = torch.Tensor([W, H]).to(xyz.device)
    shift = torch.Tensor([0.5, 0.5]).to(xyz.device)
    center = shift * wh + shift * wh * xyz
    l11 = cholesky_elements[..., 0] * cholesky_elements[..., 0]
    l12 = cholesky_elements[..., 1] * cholesky_elements[..., 0]
    l22 = cholesky_elements[..., 1] * cholesky_elements[..., 1] + cholesky_elements[..., 2] * cholesky_elements[..., 2]

    conic, radius, valid = compute_cov2d_bounds(l11, l12, l22)

    tile_min, tile_max = get_tile_bbox(center, radius, tile_bounds)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )

    depths = torch.zeros(N, device=xyz.device)
    radii = radius.to(torch.int32)
    return center, depths, radii, conic, tile_area


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

def rasterize_gaussians_sum_torch(
    means2D, radii, conics, color, opacity, depths, image_width, image_height, pix_coord,
):
    # radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=image_width, height=image_height)

    # pix_coord = torch.stack(torch.meshgrid(torch.arange(image_width), torch.arange(image_height), indexing='xy'), dim=-1).to(means2D.device) # mind
    channel_num = color.shape[-1]
    render_color = torch.ones(*pix_coord.shape[:2], channel_num).to(means2D.device)
    render_depth = torch.zeros(*pix_coord.shape[:2], 1).to(means2D.device)
    render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to(means2D.device)

    TILE_SIZE = 128
    for h in range(0, image_height, TILE_SIZE):
        h_end = min(h + TILE_SIZE, image_height)
        tile_height = h_end - h  # Actual height of the tile
        for w in range(0, image_width, TILE_SIZE):
            w_end = min(w + TILE_SIZE, image_width)
            tile_width = w_end - w  # Actual width of the tile

            # check if the rectangle penetrate the tile
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
            
            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            tile_coord = pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
            # sorted_depths, index = torch.sort(depths[in_mask])
            sorted_means2D = means2D[in_mask]#[index]
            #sorted_cov2d = cov2d[in_mask][index] # P 2 2
            # sorted_conic = sorted_cov2d.inverse() # inverse of variance
            sorted_conic = conics[in_mask]
            sorted_opacity = opacity[in_mask]#[index]
            sorted_color = color[in_mask]#[index]
            dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
            
            gauss_weight = torch.exp(-0.5 * (
                dx[:, :, 0]**2 * sorted_conic[:, 0] 
                + dx[:, :, 1]**2 * sorted_conic[:, 2]
                + 2 * dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1]))
            
            alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.999) # B P 1
            #T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
            #acc_alpha = (alpha * T).sum(dim=1)
            #tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
            #tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
            tile_color = (alpha * sorted_color[None]).sum(dim=1)
            render_color[h:h_end, w:w_end] = tile_color.reshape(tile_height, tile_width, -1)
            #self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
            #self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
    return render_color
    # return {
    #     "render": render_color,
    #     # "depth": self.render_depth,
    #     # "alpha": self.render_alpha,
    #     # "visiility_filter": radii > 0,
    #     # "radii": radii
    # }