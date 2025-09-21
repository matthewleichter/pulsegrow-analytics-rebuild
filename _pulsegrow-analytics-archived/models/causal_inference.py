from dowhy import CausalModel

def run_causal_analysis(df):
    model = CausalModel(
        data=df,
        treatment='treatment',
        outcome='outcome',
        common_causes=['age', 'income']
    )
    identified_model = model.identify_effect()
    estimate = model.estimate_effect(identified_model, method_name="backdoor.linear_regression")
    return estimate