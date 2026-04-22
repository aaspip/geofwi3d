from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt


"""Adapted from plotting utilities of pyseistr"""




def plot3d(
    d3d: np.ndarray,
    frames: Sequence[int] | None = None,
    z: np.ndarray | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dz: float = 1,
    dx: float = 1,
    dy: float = 1,
    nlevel: int = 100,
    figsize: tuple[float, float] = (8, 6),
    ifnewfig: bool = True,
    figname: str | None = None,
    showf: bool = True,
    close: bool = True,
    ifslice: bool = True,
    ifinside: bool = False,
    topo: np.ndarray | None = None,
    Vtopo: np.ndarray | None = None,
    **kwargs: Any,
) -> Any:
    """Plot three orthogonal slices of a 3D volume.

    ``d3d`` shape is ``(nz, nx, ny)`` — depth first, then X, then Y (RGM ``vp`` layout).
    """
    nz, nx, ny = d3d.shape
    if frames is None:
        frames = [int(nz / 2), int(nx / 2), int(ny / 2)]
    if z is None:
        z = np.arange(nz) * dz
    if x is None:
        x = np.arange(nx) * dx
    if y is None:
        y = np.arange(ny) * dy

    X, Y, Z = np.meshgrid(x, y, z)
    d3d = d3d.transpose([1, 2, 0])

    barlabel = kwargs.pop("barlabel", None)

    kw: dict[str, Any] = {
        "vmin": float(d3d.min()),
        "vmax": float(d3d.max()),
        "levels": np.linspace(float(d3d.min()) - 1e-9, float(d3d.max()) + 1e-9, nlevel),
        "cmap": plt.cm.grey,
    }
    kw.update(kwargs)

    def norm(v: np.ndarray) -> np.ndarray:
        return (v - kw["vmin"]) / (kw["vmax"] - kw["vmin"])

    if "alpha" not in kw:
        kw["alpha"] = 1.0

    if not ifnewfig:
        ax = plt.gca()
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect="auto", projection="3d")
        plt.jet()

    C: Any = None
    if ifinside:
        tt = d3d[:, :, 0].transpose().copy()
        tt[0 : frames[2], frames[1] + 1 :] = False  # type: ignore[assignment]
        _ = ax.contourf(X[:, :, -1], Y[:, :, -1], tt, zdir="z", offset=Z.min(), **kw)
        _ = ax.contourf(
            X[0 : frames[2] + 1, frames[1] :, -1],
            Y[0 : frames[2] + 1, frames[1] :, -1],
            d3d[frames[1] :, 0 : frames[2] + 1, frames[0]].transpose(),
            zdir="z",
            offset=z[frames[0]],
            **kw,
        )
        tt = d3d[:, 0, :].copy()
        tt[frames[1] + 1 :, 0 : frames[0]] = False  # type: ignore[assignment]
        _ = ax.contourf(X[0, :, :], tt, Z[0, :, :], zdir="y", offset=Y.min(), **kw)
        _ = ax.contourf(
            X[0, frames[1] :, 0 : frames[0] + 1],
            d3d[frames[1] :, frames[2], 0 : frames[0] + 1],
            Z[0, frames[1] :, 0 : frames[0] + 1],
            zdir="y",
            offset=y[frames[2]],
            **kw,
        )
        tt = d3d[nx - 1, :, :].copy()
        tt[0 : frames[2], 0 : frames[0]] = False  # type: ignore[assignment]
        C = ax.contourf(tt, Y[:, -1, :], Z[:, -1, :], zdir="x", offset=X.max(), **kw)
        C = ax.contourf(
            d3d[frames[1], 0 : frames[2] + 1, 0 : frames[0] + 1],
            Y[0 : frames[2] + 1, -1, 0 : frames[0] + 1],
            Z[0 : frames[2] + 1, -1, 0 : frames[0] + 1],
            zdir="x",
            offset=x[frames[1]],
            **kw,
        )
    else:
        if topo is None:
            _ = ax.contourf(
                X[:, :, -1],
                Y[:, :, -1],
                d3d[:, :, frames[0]].transpose(),
                zdir="z",
                offset=Z.min(),
                **kw,
            )
            _ = ax.contourf(X[0, :, :], d3d[:, frames[2], :], Z[0, :, :], zdir="y", offset=Y.min(), **kw)
            C = ax.contourf(d3d[frames[1], :, :], Y[:, -1, :], Z[:, -1, :], zdir="x", offset=X.max(), **kw)
        else:
            Z_top = topo
            if Vtopo is None:
                Vp_top = d3d[:, :, frames[0]].transpose()
            else:
                Vp_top = Vtopo
            _ = ax.plot_surface(
                X[:, :, -1],
                Y[:, :, -1],
                Z_top,
                rstride=1,
                cstride=1,
                facecolors=plt.cm.jet(norm(Vp_top)),
                linewidth=0,
                edgecolor="none",
                antialiased=False,
            )
            Ztop_xmin = Z_top[0, :]
            vel = d3d[:, frames[2], :]
            mask = np.squeeze(Z[0, :, :]) > Ztop_xmin[:, None] - (z[2] - z[1])
            vel = np.where(mask, vel, np.nan)
            _ = ax.contourf(X[0, :, :], vel, Z[0, :, :], zdir="y", offset=Y.min(), **kw)
            Ztop_ymin = Z_top[:, 0]
            vel = d3d[frames[1], :, :]
            mask = np.squeeze(Z[:, -1, :]) > Ztop_ymin[:, None] - (z[2] - z[1])
            vel = np.where(mask, vel, np.nan)
            C = ax.contourf(vel, Y[:, -1, :], Z[:, -1, :], zdir="x", offset=X.max(), **kw)
            ax.yaxis.pane.set_edgecolor("w")
            ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
            ax.yaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("w")
            ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
            ax.xaxis.pane.fill = False
            ax.yaxis._axinfo["grid"].update(color="w", linestyle="")
            ax.xaxis._axinfo["grid"].update(color="w", linestyle="")
            ax.zaxis._axinfo["grid"].update(color="w", linestyle="")

    plt.gca().set_xlabel("X", fontsize="large", fontweight="normal")
    plt.gca().set_ylabel("Y", fontsize="large", fontweight="normal")
    plt.gca().set_zlabel("Z", fontsize="large", fontweight="normal")
    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())
    zmin, zmax = float(Z.min()), float(Z.max())
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    plt.gca().invert_zaxis()

    if ifslice:
        plt.plot(
            [x[frames[1]], x[frames[1]]],
            [y.min(), y.max()],
            [z.min(), z.min()],
            "b-",
            linewidth=2,
            zorder=10,
        )
        plt.plot(
            [x.min(), x.max()],
            [y[frames[2]], y[frames[2]]],
            [z.min(), z.min()],
            "b-",
            linewidth=2,
            zorder=10,
        )
        plt.plot(
            [x[frames[1]], x[frames[1]]],
            [y.min(), y.min()],
            [z.min(), z.max()],
            "b-",
            linewidth=2,
            zorder=10,
        )
        plt.plot(
            [x.min(), x.max()],
            [y.min(), y.min()],
            [z[frames[0]], z[frames[0]]],
            "b-",
            linewidth=2,
            zorder=10,
        )
        plt.plot(
            [x.max(), x.max()],
            [y[frames[2]], y[frames[2]]],
            [z.min(), z.max()],
            "b-",
            linewidth=2,
            zorder=10,
        )
        plt.plot(
            [x.max(), x.max()],
            [y.min(), y.max()],
            [z[frames[0]], z[frames[0]]],
            "b-",
            linewidth=2,
            zorder=10,
        )

    if barlabel is not None and C is not None:
        cbar = plt.gcf().colorbar(
            C,
            ax=ax,
            orientation="vertical",
            fraction=0.02,
            pad=0.1,
            label=barlabel,
        )
        cbar.locator = plt.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])

    if figname is not None:
        save_kw = {k: v for k, v in kwargs.items() if k != "cmap"}
        plt.savefig(figname, **save_kw)

    if showf:
        plt.show()
    elif close:
        plt.close()

    return C