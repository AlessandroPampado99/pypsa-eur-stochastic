"""Publication layout using verified component and delivery mappings.

Render from the saved audit without re-importing or solving the PyPSA network:
    python scripts/network_plot/integrated.py
"""
from pathlib import Path
import argparse
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as pe
import yaml

HERE = Path(__file__).resolve().parent
INK = '#293b49'
CARBON = '#a03e79'


class Drawing:
    def __init__(self, height):
        self.fig = plt.figure(figsize=(180 / 25.4, height / 25.4), facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1], xlim=(0, 180), ylim=(height, 0))
        self.ax.axis('off')
        self.nodes = {}
        self.routes = []

    def text(self, x, y, label, size=8, **kw):
        return self.ax.text(x, y, label, fontsize=size, va='center',
                            color=kw.pop('color', INK), **kw)

    def node(self, name, x, y, w, h, label, fill='white', edge='#aab5bf', bold=False):
        self.ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor=fill,
                                   edgecolor=edge, lw=.65, zorder=5))
        self.text(x, y, label, 8.5 if bold else 8, ha='center',
                  weight='bold' if bold else 'normal', zorder=6)
        self.nodes[name] = (x-w/2, y-h/2, x+w/2, y+h/2)

    def edge(self, pts, color=INK, dashed=False, source=None, target=None, z=2, arrow=True):
        # White casing separates crossings. Junction dots are added only for
        # shared branches of the SAME carrier; other crossings are not joins.
        style = '--' if dashed else '-'
        line, = self.ax.plot(*zip(*pts), color=color, lw=.8, ls=style,
                            solid_capstyle='round', zorder=z)
        line.set_path_effects([pe.Stroke(linewidth=2.1, foreground='white'), pe.Normal()])
        if arrow:
            self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>',
                                         mutation_scale=6.5, lw=.8, color=color,
                                         linestyle=style, shrinkA=0, shrinkB=0, zorder=z+.1))
        self.routes.append({'points': pts, 'source': source, 'target': target,
                            'color': color, 'carbon': dashed})

    def dot(self, x, y, color):
        self.ax.plot(x, y, 'o', ms=2.1, color=color, zorder=4)

    def validate(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for item in self.ax.texts:
            b = item.get_window_extent(renderer)
            if not self.fig.bbox.contains(b.x0, b.y0) or not self.fig.bbox.contains(b.x1, b.y1):
                raise ValueError(f'Clipped text: {item.get_text()}')
        # Straight orthogonal segments must not pass through unrelated nodes.
        violations = []
        for route in self.routes:
            for name, (x0,y0,x1,y1) in self.nodes.items():
                if name in (route['source'], route['target']):
                    continue
                for (a,b),(c,d) in zip(route['points'],route['points'][1:]):
                    if a == c and x0+.2 < a < x1-.2 and max(min(b,d),y0+.2) < min(max(b,d),y1-.2):
                        violations.append(name)
                    if b == d and y0+.2 < b < y1-.2 and max(min(a,c),x0+.2) < min(max(a,c),x1-.2):
                        violations.append(name)
        if violations:
            raise ValueError(f'Routes cross node interiors: {sorted(set(violations))}')

    def save(self, out, name):
        self.validate()
        for ext in ('pdf', 'svg', 'png'):
            self.fig.savefig(out / f'{name}.{ext}', dpi=450, facecolor='white')
        plt.close(self.fig)


def conversion_panel(d, mapping, y=0):
    c = mapping['carriers']
    def node(name,x,yy,w=30,h=10,label=None):
        item=c[name]
        d.node('a_'+name,x,y+yy,w,h,label or name,item['fill'],item['color'],name in ('Hydrogen','Methanol'))
    def edge(pts,carrier,source=None,target=None,dashed=False):
        d.edge([(x,y+yy) for x,yy in pts],CARBON if dashed else c[carrier]['color'],dashed,
               'a_'+source if source else None,'a_'+target if target else None)
    def process(name,x,yy,w,h=8):
        d.node('a_'+name,x,y+yy,w,h,name)
    d.text(5,y+5,'(a) Sources and main carrier conversions',9.5,weight='bold')
    for name,x,yy,w in [('Electricity',65,19,31),('Heat',151,19,31),
                         ('Hydrogen',65,48,31),('Methanol',151,48,31),
                         ('Methane',65,79,31),('Liquid hydrocarbons',151,79,43)]:
        node(name,x,yy,w)
    d.node('a_renewables',20,y+19,30,15,'Wind / solar\nHydro / nuclear','#e8f1f7')
    d.node('a_gas',20,y+79,30,9,'Natural gas','#e8f1f7')
    d.node('a_oil',20,y+91,30,9,'Crude oil','#e8f1f7')
    node('Biomass',20,59,30,12)
    edge([(35,19),(49.5,19)],'Electricity','renewables','Electricity')
    edge([(35,79),(49.5,79)],'Methane','gas','Methane')
    edge([(35,91),(151,91),(151,84)],'Liquid hydrocarbons','oil','Liquid hydrocarbons')
    d.text(91,y+88,'Refining',8,ha='center')
    # Heat pumps / resistive heaters are alternatives, not joint products.
    edge([(80.5,17),(135.5,17)],'Electricity','Electricity','Heat')
    d.text(108,y+12,'Heat pumps / heaters',8,ha='center')
    process('Electrolysis',65,34,28,7)
    edge([(65,24),(65,30.5)],'Electricity','Electricity','Electrolysis')
    edge([(65,37.5),(65,43)],'Hydrogen','Electrolysis','Hydrogen')
    process('Gas reforming',65,64,30,7)
    edge([(65,74),(65,67.5)],'Methane','Methane','Gas reforming')
    edge([(65,60.5),(65,53)],'Hydrogen','Gas reforming','Hydrogen')
    process('Methanol\nsynthesis',107,48,24,11)
    process('Fischer–\nTropsch',107,79,24,11)
    # A shared hydrogen trunk has explicit branch points, never mixed carriers.
    edge([(80.5,48),(86,48),(95,48)],'Hydrogen','Hydrogen','Methanol\nsynthesis')
    edge([(86,48),(86,77),(95,77)],'Hydrogen',None,'Fischer–\nTropsch')
    d.dot(86,y+48,c['Hydrogen']['color'])
    edge([(80.5,22),(107,22),(107,42.5)],'Electricity','Electricity','Methanol\nsynthesis')
    edge([(119,48),(135.5,48)],'Methanol','Methanol\nsynthesis','Methanol')
    edge([(119,79),(129.5,79)],'Liquid hydrocarbons','Fischer–\nTropsch','Liquid hydrocarbons')
    # Methanol reforming is a separate enabled reverse conversion.
    process('Reforming',108,69,26,7)
    edge([(138,53),(138,69),(121,69)],'Methanol','Methanol','Reforming')
    edge([(95,69),(83,69),(83,54),(77,54),(77,53)],'Hydrogen','Reforming','Hydrogen')
    process('To kerosene',157,63,30,7)
    edge([(157,53),(157,59.5)],'Methanol','Methanol','To kerosene')
    edge([(86,59),(136,59),(136,63),(142,63)],'Hydrogen',None,'To kerosene')
    d.dot(86,y+59,c['Hydrogen']['color'])
    edge([(157,66.5),(157,74)],'Liquid hydrocarbons','To kerosene','Liquid hydrocarbons')
    # Biomass supplies methanol directly; other biomass supplies are in (b).
    edge([(35,59),(39,59),(39,56),(130,56),(130,51),(135.5,51)],'Biomass','Biomass','Methanol')
    # Explicit power routes, kept outside synthesis input corridors.
    process('Gas turbines',37,34,25,7)
    edge([(49.5,76),(40,76),(40,37.5)],'Methane','Methane','Gas turbines')
    edge([(40,30.5),(40,22),(49.5,22)],'Electricity','Gas turbines','Electricity')
    edge([(149,43),(149,28),(84,28),(84,19),(80.5,19)],'Methanol','Methanol','Electricity')
    d.text(137,y+33,'Methanol turbine',8,ha='center')
    # Small repeated-name process nodes retain CHP coupling without long loops.
    for x,carrier,process_label in [(5,'Methane','Gas CHP'),(62,'Biomass','Biomass CHP'),(121,'Hydrogen','Fuel cell')]:
        d.node('a_card_'+carrier,x+26,y+104,52,17,'','#fafbfc','#d3dbe1')
        d.text(x+2,y+99,carrier,8,color=c[carrier]['color'],weight='bold',zorder=8)
        d.text(x+2,y+104,process_label,8,zorder=8)
        d.edge([(x+25,y+102),(x+33,y+99)],c['Electricity']['color'],source='a_card_'+carrier,target='a_card_'+carrier,z=7)
        d.edge([(x+25,y+105),(x+33,y+109)],c['Heat']['color'],source='a_card_'+carrier,target='a_card_'+carrier,z=7)
        d.text(x+34,y+99,'Electricity',8,color=c['Electricity']['color'],zorder=8)
        d.text(x+34,y+109,'Heat',8,color=c['Heat']['color'],zorder=8)
    # Carbon strip is not a group boundary. One short utilisation trunk feeds
    # the actual synthesis process nodes, separate from their energy inputs.
    d.ax.add_patch(Rectangle((5,y+116),170,18,facecolor='#faf3f7',edgecolor='none',zorder=0))
    d.node('a_capture',33,y+123,54,10,'Capture / DAC','#f4e5ed',CARBON)
    d.node('a_co2',106,y+123,37,10,'Captured CO₂','#f4e5ed',CARBON)
    d.node('a_sequester',154,y+123,42,10,'Sequestration','#f4e5ed',CARBON)
    d.edge([(60,y+123),(87.5,y+123)],CARBON,True,'a_capture','a_co2')
    d.edge([(124.5,y+123),(133,y+123)],CARBON,True,'a_co2','a_sequester')
    d.edge([(118,y+118),(118,y+86),(111,y+86),(111,y+84.5)],CARBON,True,'a_co2','a_Fischer–\nTropsch')
    d.edge([(118,y+86),(124,y+86),(124,y+54),(114,y+54),(114,y+53.5)],CARBON,True,None,'a_Methanol\nsynthesis')
    d.text(33,y+131,'DAC uses electricity and heat',8,ha='center')
    d.text(130,y+131,'Dashed lines: CO₂',8,ha='center',color=CARBON)


def delivery_panel(d, mapping, affected, y=139):
    c=mapping['carriers']
    d.text(5,y+5,'(b) Carrier-to-sector delivery',9.5,weight='bold')
    # Shared vertical trunks are carrier-specific; input ports are distinct.
    order=['Heat','Electricity','Methane','Biomass','Coal','Ammonia','Hydrogen','Methanol','Liquid hydrocarbons']
    sectors=['Buildings and services','Industry and feedstocks','Agriculture','Land transport','Shipping','Aviation']
    sy=dict(zip(sectors,[18,43,66,84,102,119]))
    cy={name:17+12*i for i,name in enumerate(order)}
    for name in order:
        d.node('b_'+name,26,y+cy[name],42,9,name,c[name]['fill'],c[name]['color'],name in ('Hydrogen','Methanol'))
    for name in sectors:
        loads=mapping['sectors'][name]['loads']
        mark=' ◇' if set(loads)&affected else ''
        label=name.replace(' and ',' and\n')+mark
        d.node('b_'+name,154,y+sy[name],42,23 if name.startswith('Industry') else 13,label,'#fff0df')
    incoming={name:[e['carrier'] for e in mapping['delivery'] if e['sector']==name] for name in sectors}
    ports={}
    for name,ins in incoming.items():
        ins.sort(key=order.index)
        for i,carrier in enumerate(ins):
            ports[carrier,name]=sy[name]+(i-(len(ins)-1)/2)*2.1
    for i,carrier in enumerate(order):
        xx=53+i*8.6
        endpoints=[(e['sector'],ports[carrier,e['sector']]) for e in mapping['delivery'] if e['carrier']==carrier]
        yy=cy[carrier]
        vals=[yy]+[p for _,p in endpoints]
        # All leaves connect to the original carrier node through this trunk.
        d.edge([(47,y+yy),(xx,y+yy)],c[carrier]['color'],'',source='b_'+carrier,arrow=False)
        if max(vals)>min(vals):
            line,=d.ax.plot([xx,xx],[y+min(vals),y+max(vals)],color=c[carrier]['color'],lw=.8,zorder=2)
            line.set_path_effects([pe.Stroke(linewidth=2.1,foreground='white'),pe.Normal()])
            d.routes.append({'points':[(xx,y+min(vals)),(xx,y+max(vals))],'source':'b_'+carrier,'target':None})
        for sector,pp in endpoints:
            d.edge([(xx,y+pp),(133,y+pp)],c[carrier]['color'],source='b_'+carrier,target='b_'+sector,z=3)
            d.dot(xx,y+pp,c[carrier]['color'])
        d.dot(xx,y+yy,c[carrier]['color'])
    d.text(5,y+131,'◇ Scenario-sensitive demand categories; not every demand in a marked sector varies.',8)


def infrastructure(d,y):
    d.ax.add_patch(Rectangle((5,y),170,22,facecolor='#f2f5f7',edgecolor='none'))
    d.text(8,y+4,'Networks and storage',8.5,weight='bold')
    d.text(8,y+12,'Electricity: grids; batteries,\nEV storage and pumped hydro',8)
    d.text(66,y+12,'Hydrogen / methane: pipelines\nand stores. Heat: tanks / pits.',8)
    d.text(128,y+12,'Methanol / liquids:\nEU pools and stores',8)
    d.text(8,y+19,'Shared-colour branches connect; crossings without a dot do not. Uniform widths are not quantitative.',8)


def render(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'figure.autolayout':False})
    mapping=yaml.safe_load((HERE/'figure_mapping.yaml').read_text())
    affected=set(json.loads((out/'uncertainty_selectors.json').read_text())['marked_load_carriers'])
    inventory=__import__('pandas').read_csv(out/'loads_inventory.csv')
    for edge in mapping['delivery']:
        assert set(edge['loads'])<=set(inventory.carrier),edge
    d=Drawing(299)
    conversion_panel(d,mapping)
    delivery_panel(d,mapping,affected,139)
    infrastructure(d,274)
    d.save(out,'system_main')
    # Standalone coordinated panels are convenient for journal-specific layout.
    d=Drawing(137);conversion_panel(d,mapping);d.save(out,'system_conversions')
    d=Drawing(160);delivery_panel(d,mapping,affected,0);infrastructure(d,135);d.save(out,'system_delivery')
    (out/'delivery_edges.json').write_text(json.dumps(mapping['delivery'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'output')
    args=parser.parse_args()
    render(args.output)
