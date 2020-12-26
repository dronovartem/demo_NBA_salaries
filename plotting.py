import plotly.express as px
import pandas as pd


"""
df here - already sorted by Salary with all required fields
"""

def plot_salaries_hist(df):
    fig = px.bar(
        df,
        x='Player', 
        y='Salary',
        template='ggplot2',
        title='Salaries of top5 closest players')
    return fig

def plot_power_angle(df):
    df = df[["Player", "Salary", "PER", "USG%", "BPM"]]

    player_skills = df.drop(['Salary'], axis=1)
    polar_df = pd.melt(player_skills, id_vars=['Player'], 
                   value_vars=player_skills.drop('Player', axis=1).columns, var_name='Skills')
    
    fig = px.line_polar(polar_df, r="value", theta="Skills", color='Player', line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    template="gridon", 
                    animation_frame="Player",
                    animation_group="Player",
                    title='Skills distribution for nearest players'
                   )
    return fig


def foo_point_leader(df):
    point_leaders = df.groupby('Player').sum().sort_values(by='Points', ascending=False).head(20).reset_index()
    point_leaders = point_leaders.drop(['weight_kg', 'weight', 'height_cm', 'birth_year'], axis=1)
    
    fig = px.bar(
        point_leaders, 
        x='Player', 
        y='Points', 
        color='Points',
        color_continuous_scale = px.colors.sequential.Reds,
        template='ggplot2',
        title='Season top20 scorers')
    return fig

def __add_cols(df, *args):
    cols_range = range(1, len(list(args)))
    col_list = [col for col in args]
    base_col = df[col_list[0]].copy()

    for i in cols_range:
        base_col += df[col_list[i]]
    return base_col

def top_20_point(df):
    point_leaders = df.groupby('Player').sum().sort_values(by='Points', ascending=False).head(20).reset_index()
    point_leaders = point_leaders.drop(['weight_kg', 'weight', 'height_cm', 'birth_year'], axis=1)
    point_leaders['Total_Attempts'] = __add_cols(point_leaders, 'Field_Goals_Attempts', 
                                           'Three_Points_Attempts', 'Free_Throws_Attempts')
    point_leaders['Total_Throws_Made'] = __add_cols(point_leaders, 'Field_Goals_Made', 
                                              'Three_Points_Made', 'Free_Throws_Made')
    point_leaders['Points_Attempt_Ratio'] = (point_leaders['Points']/point_leaders['Total_Attempts'])
    
    size=point_leaders['Games_Played']
    
    fig = px.scatter(
        point_leaders, x='Points', y='Points_Attempt_Ratio',
        color='Points_Attempt_Ratio',
        color_discrete_sequence=px.colors.sequential.Reds,
        size=size, 
        hover_name='Player',
        title='Points attempt ratio influence on game score',
        trendline='ols',
        template='gridon'
    )
    fig.update_traces(marker=dict(
        sizeref=2*max(size)/(40**2), 
        sizemode='area', 
        sizemin=0,
    ))
    fig.update_layout(
        coloraxis_colorbar_thicknessmode='pixels',
        coloraxis_colorbar_thickness=10,
    )
    fig.update_traces(
        line_dash='dot',
        line_width=4,
        line_color='red')
    
    return fig
