
def run_all_models(pipeline_functions):
    results = {}
    for name, func in pipeline_functions.items():
        try:
            results[name] = func()
        except Exception as e:
            results[name] = str(e)
    return results
