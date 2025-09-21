from dowhy import CausalModel

def run_causal_model(data):
    model = CausalModel(
        data=data,
        treatment='treatment',
        outcome='outcome',
        common_causes=['age', 'gender', 'segment']
    )
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
    return estimate