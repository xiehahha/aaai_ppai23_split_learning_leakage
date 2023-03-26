import pickle
import pandas

def result_summary(original_score, res_dummy):

  res = abs(original_score-torch.tensor(res_dummy))/abs(original_score)
  print ("L1-loss mean", sum(abs(original_score-torch.tensor(res_dummy)))/score.shape[0])
  print ("mean error rate (L1)", sum(res)/score.shape[0])
  import pandas as pd

  res_list = [t.detach().numpy().tolist() for t in res]
  s = pd.Series(res_list)
  print (s.describe())