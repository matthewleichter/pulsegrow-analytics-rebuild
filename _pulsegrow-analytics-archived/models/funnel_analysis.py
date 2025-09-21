def funnel_conversion_rate(funnel_df):
    stages = funnel_df.columns
    conversion_rates = {}
    for i in range(len(stages)-1):
        conversion_rates[f"{stages[i]}→{stages[i+1]}"] = (
            funnel_df[stages[i+1]].sum() / funnel_df[stages[i]].sum()
        )
    return conversion_rates