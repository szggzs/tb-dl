import os

from pymatgen.ext.matproj import MPRester

ms_api_token = 'fn4VuzKQJkM35SAI'
m = MPRester(ms_api_token)

entries = m.get_entries_in_chemsys(['C', 'Si'])

for e in entries:
    if os.path.exists(f'data/{e.entry_id}_{e.composition.formula}_bands.json'):
        continue
    bands = m.get_bandstructure_by_material_id(e.entry_id)
    if bands is None:
        continue
    print(e.composition.formula)
    print(e.entry_id)
    f = open(f'data/{e.entry_id}_{e.composition.formula}_bands.json', 'w')
    f.write(bands.to_json())
    f.close()
