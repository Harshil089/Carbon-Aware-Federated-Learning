"""Build an offline copy of the demonstration results."""
import csv
import json
from pathlib import Path

def write_report(out,summary,sensitivity,profiles,parameters):
    with (out/'per_round_SIMULATION.csv').open(encoding='utf-8-sig',newline='') as file:
        rows=list(csv.DictReader(file))
    data={'summary':summary,'sensitivity':sensitivity,'profiles':profiles,'parameters':parameters,
          'rows':rows,'artifact_prefix':'','offline':True}
    html=(Path(__file__).parent/'web'/'index.html').read_text(encoding='utf-8')
    html=html.replace('__BOOTSTRAP__',json.dumps(data).replace('</',r'<\/'))
    (out/'dashboard.html').write_text(html,encoding='utf-8')
