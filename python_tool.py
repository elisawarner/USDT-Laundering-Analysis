from langchain_core.tools import tool, InjectedToolArg

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
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close("all")
            return f"Plot created successfully. Image saved."

        return "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"
        