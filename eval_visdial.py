import os
import torch
import json 
from utils.metrcis import SparseGTMetrics, NDCG
from Levenshtein import ratio


output_dir = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/output/visdial'

file_paths = os.listdir(output_dir)
file_paths = list(filter(lambda f: 'part' in f , file_paths))
name = file_paths[0]
file_paths = list(map(lambda f: os.path.join(output_dir, f), file_paths))

results = {}
count = 0
for pth in file_paths:
    with open(pth, 'r') as f:
        partial_res = json.load(f)
    count += len(partial_res)
    results.update(partial_res)
    # dialogs.extend(data['dialogs'])
    os.remove(pth)


name = "".join(name.split('-')[:-1]) + '.json'
output_path = os.path.join(output_dir, name)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)


# result_path = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/output/visdial/zeroshot_visdial_after_champagne_googleflant5large_results_dstc8_beam_depth_8_lenPen_0.3.json'

# with open(result_path, 'r') as f:
#     results = json.load(f)

annos_path = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/annotations/visdial_v1.0/visdial_1.0_val.json'
dense_annos_path = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/annotations/visdial_v1.0/visdial_1.0_val_dense_annotations.json'

with open(annos_path, 'r') as f:
    data = json.load(f)['data']

all_answers   = data['answers']
all_questions = data['questions']


dialogs = data['dialogs']

dialogs_dict = {}

for dialog in dialogs:
    image_id = dialog['image_id']
    for i, turn in enumerate(dialog['dialog']):
        answer_opts = [all_answers[a] for a in turn['answer_options']]
        dialogs_dict[str(image_id) + '_' + str(i+1)] = {
            'answer_opts': answer_opts,
            'gt_index': turn['gt_index'] 
        }
        # print('bla')

with open(dense_annos_path, 'r') as f:
    dense_data = json.load(f)

dense_data = {str(d['image_id']) + '_' + str(d['round_id']): d['gt_relevance'] for d in dense_data}

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

for res_key, res in results.items():
    answer_opts = dialogs_dict[res_key]['answer_opts']
    gt_index = torch.tensor(dialogs_dict[res_key]['gt_index'])

    scores = torch.tensor([ratio(res, answer_opt) for answer_opt in answer_opts]).unsqueeze(0).unsqueeze(0)
    sparse_metrics.observe(scores, gt_index)
    if res_key in dense_data:
        gt_relevance = torch.tensor(dense_data[res_key]).unsqueeze(0)
        ndcg.observe(scores.squeeze(0), gt_relevance)
    # print('bla')

print(sparse_metrics.retrieve())
print(ndcg.retrieve())
