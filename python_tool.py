from langchain_core.tools import tool, InjectedToolArg
import pandas as pd
import matplotlib.pyplot as plt

@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code and return results.
    Has access to pandas (pd), matplotlib.pyplot (plt), and a loaded dataframe (df).
    """
    # Create a namespace with your dataframe and libraries
    namespace = {
        'pd': pd,
        'plt':plt,
        'df': pd.read_csv("aggregated_features_addresses.csv")
    }

    try:
        # Capture output
        exec(code, namespace)

        # If a plot was created, save it
        if plt.get_fignums():
            filename = f"plot_{len(plt.get_fignums())}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight", format='png')
            plt.close("all")
            return f"Plot created successfully. Saved as {filename}."

        return "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"
        