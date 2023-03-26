# a good example to get the results. 
class label_surrogate(nn.Module): # add for customerized label party
  def __init__(self, dim_x=16, flag=False):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(dim_x, 8, bias=flag),
        nn.ReLU(),
        nn.Linear(8, 1, bias=flag),
        # nn.Linear(4, 1, bias=flag),
    )
    
  def forward(self, x):
    return self.layers(x)

surrogate = label_surrogate(dim_x=16, flag=True)

IND = 0
Step = 10
num = 5
data = dataset_train.X[IND:IND+Step]
score = dataset_train.y[IND:IND+Step]


known_datas = dataset_train.X[IND+Step:IND+num+Step]
known_labels = dataset_train.y[IND+Step:IND+Step+num]
# bias seems a strong factor no matter what we choose
# we may need a regularization conerning the range of score we consider !
original_label, surrogate_score, dummy_score, label_model_p = simple_short_test(mlp_new, data, score, surrogate, 
                                                                                known_datas, known_labels, 
                                                                                op="Adam", iteration=3500, learning_rate=0.1, end_threhold=0.0001,
                                                                                lamda1=0.1, lamda3=0.1, R_predict_flag=True, lamda2=0.001, PRINT_flg=True)
