from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation


def get_label_style() -> dict:
    d_labstyle = {"size": "x-large", "weight": "semibold"}
    return d_labstyle


def setup_figure(fig):
    fig.patch.set_alpha(0)
    raise NotImplementedError


def put_param_area(fig, x_pad_inch: float = 0.05, y_pad_inch: float = 0.05):
    """
    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
    x_pad_inch : float
        Horizontal padding in inches.
    y_pad_inch : float
        Vertical padding in inches.
    """
    width, height = fig.get_size_inches()
    x_pad = x_pad_inch / width
    y_pad = y_pad_inch / height

    fig.add_artist(
        Line2D([x_pad, 1 - x_pad], [-y_pad, -y_pad], linewidth=0.5, color="k")
    )

    y_text = -y_pad - 0.05 / height
    fig.text(
        x=x_pad + 0.05 / width,
        y=y_text,
        s="[parameters]",
        ha="left",
        va="top",
        fontsize="small",
        # weight="semibold",
    )
    d_paramstyle = {
        "fontsize": "small",
        "x": x_pad + 0.9 / width,
        "y": y_text,
        "ha": "left",
        "va": "top",
    }

    return d_paramstyle


def param_val_to_str(name: str, value: float, ndigit: int = 4) -> str:
    if 1e-2 <= abs(value) <= 10000:
        val_str = f"{round(value, ndigits=ndigit)}"
    else:
        lfsn = LogFormatterSciNotation()
        val_str = lfsn(value)[1:-1]  # exclude "$"
    return "$" + name + "=" + val_str + "$"
