import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_event_distribution_hardened(users_df, events_df):
    fallback = go.Figure().update_layout(
        title="❌ Plot could not be generated due to input error"
    )

    required_user_cols = {'user_id', 'plan_type'}
    required_event_cols = {'user_id'}

    if not required_user_cols.issubset(users_df.columns):
        fallback.update_layout(title="❌ Missing columns in users_df")
        return fallback

    if not required_event_cols.issubset(events_df.columns):
        fallback.update_layout(title="❌ Missing columns in events_df")
        return fallback

    if users_df.empty or events_df.empty:
        fallback.update_layout(title="❌ One or both input DataFrames are empty")
        return fallback

    try:
        df = users_df.copy()
        df['plan_type'] = df['plan_type'].astype(str).str.lower()
        df['is_paid'] = (df['plan_type'] == 'paid').astype(int)

        event_counts = events_df['user_id'].value_counts().to_dict()
        df['event_count'] = df['user_id'].map(event_counts).fillna(0).astype(int)

        if not pd.api.types.is_numeric_dtype(df['event_count']):
            fallback.update_layout(title="❌ event_count is not numeric")
            return fallback

        fig = px.histogram(
            df,
            x='event_count',
            color='is_paid',
            barmode='overlay',
            title='Event Count Distribution: Free vs Paid Users',
            labels={'event_count': 'Events', 'is_paid': 'Paid (1) vs Free (0)'}
        )

        if hasattr(fig, "to_dict"):
            return fig
        else:
            fallback.update_layout(title="❌ Invalid Plotly figure object")
            return fallback

    except Exception as e:
        return fallback.update_layout(title=f"❌ Exception: {str(e)}")
