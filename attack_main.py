# very raw single_data/target and no list things
def square_error(pred, target):
  return torch.sum((pred-target)**2)
loss_function = square_error

def mean_error(pred, target): # do not use L1-error
  return torch.sum(abs(pred-target))


# define distance with cosine for triplet distance loss
cos = torch.nn.CosineSimilarity(dim=0)



# result analyze 

def result_summary(original_score, res_dummy):

  res = abs(original_score-torch.tensor(res_dummy))/abs(original_score)
  print ("L1-loss mean", sum(abs(original_score-torch.tensor(res_dummy)))/score.shape[0])
  print ("mean error rate (L1)", sum(res)/score.shape[0])

  return sum(abs(original_score-torch.tensor(res_dummy)))/score.shape[0], sum(res)/score.shape[0]

# use a json file to log it
dic = {}
loss_function = square_error
predic_loss_func = square_error

cos = torch.nn.CosineSimilarity(dim=0)

def simple_short_test(model, datas, labels,
                      label_model, # surrogate model
                      known_datas=None, known_labels=None, # known data points
                      iteration=2000, end_threhold=0.001, op='Adam', learning_rate=1,
                      semi_flag=True, lamda1=0.05, lamda3=0.01,
                      R_predict_flag=True, lamda2=0.05, 
                      Tri_flag=False, k=0, lamda4=0.05,
                      PRINT_INT=False,
                      PRINT_flg=False):
  
  torch.manual_seed(66)


  original_dy_dx_list = []
  dummy_score_list = []

  for (data, score) in zip(datas, labels):
    data = data.reshape(1, -1)
    score = score.reshape(1, -1)

    # compute the loss to be sent g
    f_embding = model.user(data.float())
    out = model.label(f_embding)
    score = score.reshape((1,-1))
    y = loss_function(out,  score.reshape((1, -1)))
    dy_dx = torch.autograd.grad(y.float(), f_embding) # g
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
  

    original_dy_dx_list.append(original_dy_dx)

    dummy_score = torch.randn(score.size()).requires_grad_(True)
    dummy_score_list.append(dummy_score)
  


  # if there are less data, select all the data, otherwise randomly select few 3 tuples for penalty
  trip_list = list(combinations(range(datas.shape[0]), 3))
  original_label = list((_.detach().clone() for _ in dummy_score_list))

  torch.manual_seed(42)
  
  if PRINT_flg:
    print("initila surrogate model parameters", list(label_model.parameters()))
  
  # step2b: start the reconstruction process

  if op=="Adam":
    optimizer = torch.optim.Adam(list(label_model.parameters())+dummy_score_list, lr=learning_rate)
  elif op=="SGD":
    optimizer = torch.optim.SGD(list(label_model.parameters())+dummy_score_list, lr=0.01, momentum=0.9)
  else:
    optimizer = torch.optim.LBFGS(list(label_model.parameters())+dummy_score_list, lr=learning_rate)


  L1_loss = []
  L1_error_rate = []

  for iters in range(iteration):

    # print ("===========current parameters of iterations", iters, " ============")
    # for param_group in optimizer.param_groups:
    #   print (list(param_group.values())[0]) 

    # print ("current gradient of iterations", iters)
    # print ("model grad", list(label_model.parameters())[0].grad)
    # for i in range(len(dummy_score_list)):
    #   print ("label grad", dummy_score_list[i].grad)
    if PRINT_INT:
      print ("============current gradient of iterations", iters, " ============")
      print ("GT labels", labels)
      print("original score", model(datas.float()).reshape(1, -1))
      print ("dummy labels", torch.tensor(dummy_score_list))
      print("surrogate score ", label_model(model.user(datas.float())).reshape(1, -1))

    def closure():
      optimizer.zero_grad()
      agg_loss = 0

      # all the data point loss     
      for (data, score, dummy_score, original_dy_dx) in zip(datas, labels, dummy_score_list, original_dy_dx_list): 

        f_embding = model.user(data.float())
        pred = label_model(f_embding) # dummy prediction
        dummy_loss = loss_function(pred, dummy_score)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, f_embding, create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()

        # this regularization to make the surrogate model behaves as normal 
        if R_predict_flag: # print the predict_loss scale
          predict_loss = predic_loss_func(pred, dummy_score)
          grad_diff += lamda2*predict_loss
        agg_loss+=grad_diff



      if Tri_flag:
        for tri in trip_list:
          # output embedding is similar (the output of user model)
          flag = datas[tri[0]]
          c1 = datas[tri[1]]

          c2 = datas[tri[2]]

          sim1 = cos(flag, c1)
          sim2 = cos(flag, c2) # cosine 

          #distance_1 = abs(pred[tri[0]]-pred[tri[1]])[0]
          #distance_2 = abs(pred[tri[0]]-pred[tri[2]])[0]
          distance_1 = abs(dummy_score_list[tri[0]]-dummy_score_list[tri[1]])
          distance_2 = abs(dummy_score_list[tri[0]]-dummy_score_list[tri[2]])
      
          if sim1 < sim2:
            distance_loss = torch.max(torch.tensor(0), k+distance_2-distance_1) #k>0
          else:
            distance_loss = torch.max(torch.tensor(0), k+distance_1-distance_2)
          
          #print ("distance_loss", distance_loss)
          #print ("agg_loss", agg_loss)
          agg_loss += lamda4*distance_loss[0][0]


      if known_datas!=None:

        known_original_dy_dx_list = []
    
        for (data, score) in zip(known_datas, known_labels):
          data = data.reshape(1, -1)
          score = score.reshape(1, -1)

          # compute the loss to be sent g
          f_embding = model.user(data.float())
          out = model.label(f_embding)
          score = score.reshape((1,-1))
          y = loss_function(out,  score.reshape((1, -1)))
          dy_dx = torch.autograd.grad(y.float(), f_embding) # g
          known_original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        

          known_original_dy_dx_list.append(known_original_dy_dx)

        # this regularization forces the model surrogate to behave like the original model
        # adjust the MODEL
        # use the gradient loss with GT label

        for (data, score, known_original_dy_dx) in zip(known_datas, known_labels, known_original_dy_dx_list): 
          f_embding = model.user(data.float())
          pred = label_model(f_embding) # dummy prediction
          dummy_loss = loss_function(pred, score)
          dummy_dy_dx = torch.autograd.grad(dummy_loss, f_embding, create_graph=True)
          
          #label_loss = loss_function(dummy_score, score)
          grad_diff = 0
          for gx, gy in zip(dummy_dy_dx, known_original_dy_dx): 
              grad_diff += ((gx - gy) ** 2).sum()
          agg_loss+=lamda1 *grad_diff

          predict_loss_known = predic_loss_func(pred, score)
          agg_loss += lamda3*predict_loss_known


      # add semi-supervised data point (the prediction) 

      agg_loss.backward()
      
      return grad_diff

    optimizer.step(closure)

    
    # print (iters, "%.4f" % closure().item())

    if closure().item() < end_threhold:
      break

    norm_loss, error_rate = result_summary(labels, dummy_score_list)
    L1_loss.append(norm_loss)
    L1_error_rate.append(error_rate)

    if iters % 50 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
  
  if PRINT_flg:
    print (original_dy_dx)

    print("GT (ground truth) original score", labels)
    print("initial dummy score", original_label)
    print("end dummy score", dummy_score_list)

    print ("ground truth label party model", list(model.label.parameters()))
    print("end surrogate label_model", list(label_model.parameters()))

    print("original prediction score", model(datas.float()))
    print("surrogate prediction score ", label_model(model.user(datas.float())))

    
  return labels, label_model(model.user(data.float())), dummy_score_list, label_model, L1_loss, L1_error_rate