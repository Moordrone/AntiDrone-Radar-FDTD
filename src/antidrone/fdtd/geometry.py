def drone_box_like(fields, center, size, eps_r=3.0, sigma=0.0):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = max(cx - sx//2, 0), min(cx + sx//2, fields.g.nx)
    y0, y1 = max(cy - sy//2, 0), min(cy + sy//2, fields.g.ny)
    z0, z1 = max(cz - sz//2, 0), min(cz + sz//2, fields.g.nz)
    fields.eps_r[x0:x1, y0:y1, z0:z1] = float(eps_r)
    fields.sigma[x0:x1, y0:y1, z0:z1] = float(sigma)
