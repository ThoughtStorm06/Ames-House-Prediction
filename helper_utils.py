import kagglehub as kh
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def data_split(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    target="SalePrice"
    test_size=0.2
    random_state= 42
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def _ensure_axes_array(axes):
    """Normalize axes from plt.subplots to always be a flat iterable."""
    if not hasattr(axes, "flatten"):
        return np.array([axes])
    return axes


def boxplot_grid(df, plot_columns, target="SalePrice", ncols=3,
                 save_path=None, suptitle="Box Plots"):
    """Grid of box-plots for categorical features vs a continuous target.

    Args:
        df: DataFrame containing all columns.
        plot_columns: list of categorical column names.
        target: continuous column to plot on the y-axis.
        ncols: columns per row in the subplot grid.
        save_path: if given, save the figure to this path.
        suptitle: figure super-title.
    """
    nrows = (len(plot_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 8))
    axes = _ensure_axes_array(axes)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(plot_columns):
            ax.set_visible(False)
            continue
        sns.boxplot(data=df, y=target, hue=plot_columns[i], gap=0.2, ax=ax)
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
        ax.set_xlabel(plot_columns[i])
        ax.set_title(f"Dispersion of {target} for categories in {plot_columns[i]}")

    plt.suptitle(suptitle, fontsize=20, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def bar_chart_grid(df, plot_columns, ncols=3,
                   save_path=None,
                   suptitle="Distribution of Data Across Categories"):
    """Grid of bar charts showing value-counts for each column.

    Args:
        df: DataFrame containing all columns.
        plot_columns: list of categorical column names.
        ncols: columns per row in the subplot grid.
        save_path: if given, save the figure to this path.
        suptitle: figure super-title.
    """
    nrows = (len(plot_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 8))
    axes = _ensure_axes_array(axes)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(plot_columns):
            ax.set_visible(False)
            continue
        counts = df[plot_columns[i]].value_counts()
        bars = ax.bar(counts.index.astype(str), counts.values)
        ax.bar_label(bars, padding=2)
        ax.set_xlabel(plot_columns[i])
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of categories in {plot_columns[i]}")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(suptitle, fontsize=20, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def correlation_table(df, features, target="SalePrice"):
    """Pearson, Spearman and Kendall correlations of *features* against *target*.

    Returns the correlation DataFrame (target row excluded) and calls
    ``display()`` so the result renders in Jupyter.
    """
    cols = list(dict.fromkeys(features + [target]))  # unique, order-preserved
    corr_df = pd.DataFrame({
        "Pearson":  df[cols].corr(method="pearson")[target],
        "Spearman": df[cols].corr(method="spearman")[target],
        "Kendall":  df[cols].corr(method="kendall")[target],
    }).drop(target)
    #display(corr_df)
    return corr_df


def scatter_subplot(df, source: list, target: str,
                    save_path=None, alpha=0.2):
    """Grid of scatter plots for numeric features vs a target variable.

    Args:
        df: DataFrame containing all columns.
        source: list of numeric column names to plot on the x-axis.
        target: column name for the y-axis.
        save_path: if given, save the figure to this path.
        alpha: transparency for scatter points.
    """
    row = (len(source) + 2) // 3
    column = 3

    fig, axes = plt.subplots(
        row,
        column,
        figsize=(15, row * 4)
    )
    axes = _ensure_axes_array(axes)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(source):
            ax.set_visible(False)
            continue

        ax.scatter(df[source[i]], df[target], alpha=alpha)
        ax.set_title(source[i] + " vs " + target)
        ax.set_xlabel(source[i])
        ax.set_ylabel(target)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def missing_summary(df, col_name="Empty rows"):
    """Missing-value counts per column as a DataFrame."""
    return pd.DataFrame(df.isna().sum(), columns=[col_name])