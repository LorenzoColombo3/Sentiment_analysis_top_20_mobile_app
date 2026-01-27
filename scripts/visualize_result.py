import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import urllib.parse
import sys

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
except NameError:
    pass

FILE_GLOBAL = 'word_freq_results_parquet.csv'
FILE_APPS = 'app_insights_drilldown.csv'

if not os.path.exists(FILE_GLOBAL) or not os.path.exists(FILE_APPS):
    print(f"ERRORE: File CSV mancanti nella cartella:\n{os.getcwd()}")
    sys.exit(1)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

print("1. Caricamento e Pulizia Dati")
df_global = pd.read_csv(FILE_GLOBAL)
df_apps = pd.read_csv(FILE_APPS)
df_apps['app_name'] = df_apps['app_name'].apply(lambda x: urllib.parse.unquote(str(x)))
target_apps_candidates = ['Facebook', 'Candy Crush Saga', 'Dropbox', 'WhatsApp', 'Microsoft Word', 'Spotify']
available_apps = df_apps['app_name'].unique()
selected_apps = [app for app in target_apps_candidates if any(app.lower() in s.lower() for s in available_apps)][:4]

print("2. Generazione Grafico 1: Global Ecosystem")
top_pos = df_global[df_global['rating'] == 5.0].head(10).copy()
top_neg = df_global[df_global['rating'] == 1.0].head(10).copy()
top_neg['count'] = top_neg['count'] * -1

df_diverging = pd.concat([top_neg.iloc[::-1], top_pos.iloc[::-1]])
colors = ["#005eff" if x > 0 else '#d62728' for x in df_diverging['count']]

plt.figure(figsize=(14, 10))
bars = plt.barh(df_diverging['bigram'], df_diverging['count'], color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title('Global Ecosystem: Pain Points vs Success Factors', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Frequency (Negative < 0 | Positive > 0)')
sns.despine(left=True, bottom=True)

for bar in bars:
    w = bar.get_width()
    plt.text(w + (2000 if w > 0 else -2000), bar.get_y() + bar.get_height()/2, 
             f'{int(abs(w)):,}', va='center', ha='left' if w>0 else 'right', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('1_global_sentiment.png')

def create_drilldown_chart(rating, title, color_hex, filename):
    print(f"3. Generazione Grafico: {filename}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)
    
    for i, app_name_partial in enumerate(selected_apps):
        ax = axes.flatten()[i]
        full_name = next((s for s in available_apps if app_name_partial.lower() in s.lower()), app_name_partial)
        
        subset = df_apps[
            (df_apps['app_name'] == full_name) & 
            (df_apps['rating'] == rating)
        ].sort_values('count', ascending=False).head(5)
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            continue

        sns.barplot(data=subset, x='count', y='bigram', ax=ax, color=color_hex)
        ax.set_title(full_name, fontsize=16, fontweight='bold', color='#333333')
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y()+p.get_height()/2), 
                        xytext=(5, 0), textcoords='offset points', va='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)

create_drilldown_chart(1.0, 'Drill-Down: Why Users Complain? (Negative Drivers)', '#d62728', '2_app_pain_points.png')

create_drilldown_chart(5.0, 'Drill-Down: Why Users Love It? (Killer Features)', "#0044ff", '3_app_success_factors.png')

print("finito generazionte grafici!")