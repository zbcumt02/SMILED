from Main import *

filename = 'results/CAL500_20260524_172552.pt'
all_tensor = torch.load(filename)

all_targets, all_preds, all_scores = (all_tensor['all_targets'],
                                      all_tensor['all_preds'],
                                      all_tensor['all_scores'])

scores_add = 10
all_targets[all_targets == 0] = -1
all_preds[all_preds == 0] = -1
all_scores = all_scores.clone()
min_values = all_scores.min(dim=0, keepdim=True)[0] + all_scores.min(dim=0, keepdim=True)[0] + scores_add
max_values = all_scores.max(dim=0, keepdim=True)[0] + all_scores.min(dim=0, keepdim=True)[0] + scores_add
all_scores = (all_scores + all_scores.min(dim=0, keepdim=True)[0] + scores_add - min_values) / (
        max_values - min_values)

hamming = Hamming_loss(all_targets, all_preds)
ranking_loss = Ranking_loss(all_scores, all_targets)
avg_precision = Average_precision(all_scores, all_targets)
coverage = Coverage(all_scores, all_targets)
one_error = One_error(all_scores, all_targets)

print("Total:"
      "Hamming Loss: %.5f\n"
      "Ranking Loss: %.5f\n"
      "Average Precision: %.5f\n"
      "Coverage: %.5f\n"
      "One Error: %.5f" % (hamming, ranking_loss, avg_precision, coverage, one_error))
