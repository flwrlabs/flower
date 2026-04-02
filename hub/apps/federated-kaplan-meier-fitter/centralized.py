import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons

if __name__ == "__main__":
    X = load_waltons()
    fitter = KaplanMeierFitter()
    fitter.fit(X["T"], X["E"])
    print("Survival function")
    print(fitter.survival_function_)
    print("Mean survival time:")
    print(fitter.median_survival_time_)
    fitter.plot_survival_function()
    plt.title("Survival function of fruit flies (Walton's data)", fontsize=16)
    plt.savefig("./_static/survival_function_centralized.png", dpi=200)
    print("Centralized survival function saved.")
