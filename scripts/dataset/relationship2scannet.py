import json
import pathlib
import argparse

parser = argparse.ArgumentParser(description='Filter 3DSSG relationships to only include those that are in the ScanNet labels.')
parser.add_argument('--path', type=pathlib.Path, required=True, help='3RScan directory')
args = parser.parse_args()

object_json_path = args.path / "3DSSG/objects.json"
rel_json_path = args.path / "3DSSG_subset/relationships.json"
scannet_mapping = args.path / "3DSSG_subset/3dssg_to_scannet.json"

with open(object_json_path, 'r') as f:
    object_data = json.load(f)
with open(rel_json_path, 'r') as f:
    rel_data = json.load(f)
with open(scannet_mapping, 'r') as f:
    scannet_mapping = json.load(f)

categories2ID = {}
categories = scannet_mapping['ScanNet_list']
for i, category in enumerate(categories):
    categories2ID[category] = i

OBJ2CLASS = {}
for scan in object_data["scans"]:
    new_scan = scan["scan"]
    id_to_label = {}
    for obj in scan["objects"]:
        if scannet_mapping['3DSSG2NYUv2'][obj["label"]] not in categories2ID:
            id_to_label[int(obj["id"])] = -1
        else:
            id_to_label[int(obj["id"])] = categories2ID[scannet_mapping['3DSSG2NYUv2'][obj["label"]]]
    OBJ2CLASS[new_scan] = id_to_label

filtered_relationships = {"scans": []}

for scan in rel_data['scans']:
    scan_id = scan['scan']
    relationships = scan['relationships']
    filtered_relationships['scans'].append({
        "scan": scan_id,
        "relationships": []
    })
    for rel in relationships:
        if OBJ2CLASS[scan_id][rel[0]] != -1 and OBJ2CLASS[scan_id][rel[1]] != -1 and rel[3] in scannet_mapping['ScanNet_rel']:
            filtered_relationships['scans'][-1]['relationships'].append([rel[0], rel[1], scannet_mapping['ScanNet_rel'].index(rel[3]), rel[3]])

output_path = args.path / "3DSSG_subset/relationships20.json"
with open(output_path, 'w') as f:
    json.dump(filtered_relationships, f)