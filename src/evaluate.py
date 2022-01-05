#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, json
from math import log10

import numpy as np

from sklearn.cluster import DBSCAN

import torch

from config import config, get_model
from session import Session
from sut.sut_sbst import move_road, frechet_distance

def evaluate_session(session):
  print("Reporting on session {}".format(session.id))
  print()

  # Load the session training data.
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    X = np.load(f)
    Y = np.load(f)

  t = [[100.0, 100.0], [100.0, 98.94274979309532], [99.9886526387778, 97.89042273490794], [99.96513502692241, 96.84416897891343], [99.92928353240157, 95.80300200366852], [99.8822483086908, 94.7677446985842], [99.82370209172556, 93.73642402077391], [99.75479503498136, 92.70986285964865], [99.67371772760399, 91.68937500071624], [99.58112942697211, 90.6728237690579], [99.47916680800455, 89.66086842205227], [99.36487030637151, 88.65399985579626], [99.24103585437051, 87.6507406527497], [99.10684056259053, 86.65224096638816], [98.96228443103162, 85.65850079671164], [98.80720382766145, 84.66853362227688], [98.64258527392333, 83.68217581105158], [98.46859240184948, 82.700413884479], [98.28391142593212, 81.72143843170493], [98.09084265312231, 80.74689523155128], [97.88724940850125, 79.77612502663939], [97.67494110292319, 78.8078140314614], [97.45424500045269, 77.84393528890385], [97.22384731558188, 76.8826793881125], [96.98489820178638, 75.92486921849834], [96.73723402703382, 74.96951825861808], [96.48200494479983, 74.01744939788269], [96.21723791219782, 73.06898990035675], [95.94572886152531, 72.12266234850014], [95.66550474989577, 71.17879400637746], [95.37656557730922, 70.23738487398867], [95.08104801868447, 69.29909420871248], [94.77763828851364, 68.36211259969463], [94.46650001882904, 67.4274265683784], [94.14878336310623, 66.49585900417476], [93.82416105728063, 65.56543686419718], [93.49181021194126, 64.63731030192122], [93.15354023794232, 63.71016553183902], [92.80754172442964, 62.78531633945843], [92.45578771428971, 61.86243546071488], [92.09811457549023, 60.9405363741651], [91.73353578658794, 60.01978271184138], [91.3630378690261, 59.10001084171143], [90.98678445483702, 58.18220728521851], [90.6044482799561, 57.264398999476086], [90.21734312989119, 56.34839539533841], [89.82431885116672, 55.433373583394506], [89.42619833319367, 54.51818341016881], [89.02314520800435, 53.60381139710459], [88.61417295415546, 52.69042117623414], [88.20125461453355, 51.777685483492874], [87.78324003566307, 50.864781429469815], [87.36111573898725, 49.951545382132686], [86.93405883509514, 49.03912749495703], [86.50387873484098, 48.126213982435], [86.06860239533822, 47.21313210863118], [85.62921633803013, 46.29971824151329], [85.18687071639228, 45.38679527049229], [84.7402517449168, 44.47255378471395], [84.29050957707926, 43.55781667358924], [83.83764421287967, 42.64258393711815], [83.38165565231799, 41.72685557530069], [82.92238026336194, 40.8096450666936], [82.45998167804383, 39.891938932740125], [81.99544641780693, 38.97357354140799], [81.5284472185866, 38.05257584981067], [81.05848845503652, 37.13206905431024], [80.58441997368111, 36.21123026549574], [80.1073919279959, 35.290882372778114], [79.62559490712673, 34.372339161665224], [79.14017906454912, 33.45642352156803], [78.64900772534423, 32.54247619510785], [78.15142163213345, 31.632633857203516], [77.6485709383923, 30.727719397266], [77.13946912267753, 29.82789644732756], [76.6223067741349, 28.934478792896044], [76.09790678217539, 28.046316280495958], [75.56577352145263, 27.1665321064893], [75.02590699196662, 26.29512627087611], [74.47633415083084, 25.432426037720944], [73.91721863007757, 24.579417928467123], [73.34872406173913, 23.737088464557843], [72.76986392437226, 22.905601278025415], [72.17997896059829, 22.087093043788656], [71.57874190635262, 21.279590718961046], [70.96680728976443, 20.487040389315638], [70.3418748038826, 19.707796276030464], [69.70427171277179, 18.943831421992073], [69.0539980164319, 18.19514582720045], [68.3892443040088, 17.463053277163453], [67.71116072897797, 16.748376661292042], [67.0187607698962, 16.051279611618483], [66.31023501590924, 15.373075913650666], [65.58755650990364, 14.713438303323983], [64.84809295161418, 14.07483071962183], [64.09283086248413, 13.457089530511936], [63.31996083165929, 12.861528521502109], [62.5303057485506, 12.286997539116797], [61.723206355779396, 11.735633258274873], [60.90113132157857, 11.203985218549604], [60.063257756537155, 10.693203573416596], [59.21090417541249, 10.199014973038203], [58.34538436371242, 9.723228828268631], [57.46818046822658, 9.262558310713558], [56.579128856922665, 8.81601689892969], [55.680038940654924, 8.382290807409191], [54.77173360883428, 7.960229882676529], [53.85503575087171, 7.548683971256139], [52.93175477762144, 7.146339287640146], [52.00074053560794, 6.752372942417622], [51.065611846539596, 6.364157364572861], [50.12620507838415, 5.98070603266261], [49.1825202311416, 5.602018946686854], [48.23718960507715, 5.225632167662198], [47.29004956815844, 4.850559174145388], [46.34175937776419, 4.474663291217638], [45.39445570881322, 4.098603776257576], [44.44863418665185, 3.719257432903163], [43.50626785416664, 3.336296997089775], [42.56604292584973, 2.947913057963234], [41.63075533399858, 2.552628197983367], [40.700405078613215, 2.1504424171502023], [39.77746082792653, 1.7379052550370488], [38.861099692527524, 1.3161668651194702], [37.95280381920587, 0.8819404190031293], [37.05454625084805, 0.4348986526234029], [36.16599972338949, -0.026931476906199237], [35.2891372797167, -0.5038772336503001], [34.42461817720837, -0.9980752925276732], [33.57244241586446, -1.5095256535383328], [32.734255774506906, -2.040528623633378], [31.90923536372479, -2.5899340493372733], [31.09754481555035, -3.1567554092067525], [30.297211087097082, -3.740665439177235], [29.50856144242961, -4.339691096362159], [28.73060936010465, -4.953668748729285], [27.96319120808991, -5.583584917721851], [27.205811361038982, -6.226316406977801], [26.457483297508674, -6.881699584464812], [25.718207017498983, -7.5497344501829105], [24.98715963159887, -8.229270850656519], [24.264341139808423, -8.920308785885666], [23.548105763305642, -9.620547948919224], [22.838289870058276, -10.330974861200474], [22.13522072413093, -11.049616479842896], [21.43692528263705, -11.776145540781854], [20.744553699052233, -12.509739154606422], [20.056132930489866, -13.250070057252046], [19.371826608982374, -13.996151727275404], [18.690811845118702, -14.746834011200974], [18.013088638898893, -15.502116909028771], [17.336847579468724, -16.26068663525092], [16.663238820303732, -17.021720300456465], [15.990125686485086, -17.785877162024065], [15.317999074109736, -18.550197655624004], [14.646695351145326, -19.31566830269948], [13.974405106737663, -20.08097531774264], [13.301128340886748, -20.84611870075352], [12.62620579621391, -21.60896177681329], [11.949637472719132, -22.36950454592194], [11.270600480991476, -23.126596854603946], [10.587944667555377, -23.881061592270285], [9.902160928507712, -24.62993919459113], [9.211112588929694, -25.37388891894514], [8.516936323740126, -26.112251507953744], [7.816836200641518, -26.84354954407671], [7.110975851666154, -27.566796505870812], [6.398205123338542, -28.282815282747038], [5.678851279723176, -28.989632831818795], [4.953077952852439, -29.686262631642904], [4.21776194636422, -30.373200307565668], [3.4752035672096184, -31.048800080765233], [2.7234297725021293, -31.712734687176976], [1.961617672830812, -32.36385397332538], [1.189767268195638, -33.00215793921035], [0.4070556691856666, -33.62649643135646], [-0.38635349216684745, -34.235882928320336], [-1.1922696267160546, -34.829003644594195], [-2.0098698450510852, -35.40700873365358], [-2.83981340455054, -35.96776152057967], [-3.6839097160686407, -36.50994821986464], [-4.542322411637713, -37.03455535295177], [-5.413737705749867, -37.53977350898674], [-6.301115162734902, -38.025111791872746], [-7.204454782592876, -38.490570201609785], [-8.12342930125908, -38.934175695311296], [-9.059025240176865, -39.35576464094504], [-10.011078967313892, -39.754350517067735], [-10.981563525556695, -40.12960605961479], [-11.970315282872946, -40.48054474714294], [-12.977170607230406, -40.80618005820895], [-14.00410254151555, -41.10618472874819], [-15.050947453696125, -41.37957223731743], [-16.117705343772045, -41.62634258391665], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
  p = move_road(t, 0, 0)
  print(p[:5])
  raise SystemExit

  # Report positive tests etc.
  #report(X, Y)

  # Report analyzer performance over the experiment.
  #test_analyzer()
  #analyzer_performance()
  report_median(X, Y)
  cluster(X, Y)
  return

  # Generate new samples to assess quality visually.
  total = 1000
  print("Generating {} new tests...".format(total))
  new_tests = model.generate_test(total)
  print("Covariance matrix of the generated tests:")
  print(np.cov(new_tests, rowvar=False))
  for n in range(30):
    #view_test(new_tests[n,:])
    save_test(new_tests[n,:], "eval_{}".format(n + 1))
  fitness = model.predict_fitness(new_tests)
  total_predicted_positive = sum(fitness >= model.sut.target)[0]
  print("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

def report_median(X, Y):
  # Get the positive tests.
  positive_tests = X[(Y >= model.sut.target).reshape(-1)]
  # Convert to plane points.
  positive_roads = [model.sut.test_to_road_points(positive_tests[n]) for n in range(len(positive_tests))]
  # Normalize.
  positive_roads = [move_road(P, 0, 0) for P in positive_roads]
  # Convert to an element of a high-dimensional vector space.
  #positive_roads = [np.array(P).reshape(-1) for P in positive_roads]

  # Compute pairwise distances.
  distances = []
  for i in range(len(positive_roads)):
    for j in range(i + 1, len(positive_roads)):
      d = frechet_distance(positive_roads[i], positive_roads[j])
      #d = np.linalg.norm(positive_roads[i] - positive_roads[j])
      distances.append(d)

  print("Mean: {}".format(np.mean(distances)))
  print("Std: {}".format(np.std(distances)))
  print("Median: {}".format(np.median(distances)))

def cluster(X, Y):
  # Get the positive tests.
  positive_tests = X[(Y >= model.sut.target).reshape(-1)]
  # Convert to plane points.
  positive_roads = [model.sut.test_to_road_points(positive_tests[n]) for n in range(len(positive_tests))]
  # Convert to normalized angle format.
  positive_roads = [move_road(P, 0, 0) for P in positive_roads]
  # Convert to an element of a high-dimensional vector space.
  positive_roads = [np.array(P).reshape(-1) for P in positive_roads]

  clustering = DBSCAN(eps=1, min_samples=1).fit(positive_roads)
  print(clustering.labels_)

def report(X, Y):
  total = Y.shape[0]
  total_positive = sum(Y >= model.sut.target)[0]
  print("{}/{} ({} %) tests are positive.".format(total_positive, total, round(total_positive/total*100, 1)))
  total_noninitial_positive = sum(Y[session.random_init:,] >= model.sut.target)[0]
  print("{}/{} ({} %) non-initial tests are positive.".format(total_noninitial_positive, total, round(total_noninitial_positive/total*100, 1)))
  avg = np.mean(Y)
  print("The test suite has average fitness {}.".format(avg))
  avg_initial = np.mean(Y[:session.random_init])
  print("The initial tests have average fitness {}.".format(avg_initial))
  avg_noninitial = np.mean(Y[session.random_init:])
  print("The noninitial tests have average fitness {}.".format(avg_noninitial))
  window = 10
  print("Moving averages with window size {}.".format(window))
  mavg = []
  for n in range(window - 1, session.N_tests - 1):
    mavg.append(round(np.mean(Y[n - window + 1:n + 1]), 2))
  print(mavg)

def test_analyzer():
  N = 50
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    data_X = np.load(f)
    data_Y = np.load(f)

  with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
    X_valid = torch.from_numpy(np.load(f)).float().to(model.device)
    Y_valid = torch.from_numpy(np.load(f)).float().to(model.device)

  K = 20
  X_valid = X_valid[:K]
  Y_valid = Y_valid[:K]

  model.initialize()

  train_settings = {"analyzer_epochs": 5}
  for n in range(session.random_init, session.N_tests):
    X = data_X[:n]
    Y = data_Y[:n]
    model.train_analyzer_with_batch(X, Y, train_settings=train_settings)
    set_X = torch.from_numpy(X).float().to(model.device)
    set_Y = torch.from_numpy(Y).float().to(model.device)
    weights = model.analyzer.weights(set_Y)
    loss1 = model.analyzer.analyzer_loss(set_X, set_Y, weights)
    weights = model.analyzer.weights(Y_valid)
    loss2 = model.analyzer.analyzer_loss(X_valid, Y_valid, weights)
    print("{}: loss on training: {}, loss on validation: {}".format(n, loss1, loss2))

  set_X = torch.from_numpy(data_X).float().to(model.device)
  set_Y = torch.from_numpy(data_Y).float().to(model.device)
  weights = model.analyzer.weights(set_Y)
  loss1 = model.analyzer.analyzer_loss(set_X, set_Y, weights)
  weights = model.analyzer.weights(Y_valid)
  loss2 = model.analyzer.analyzer_loss(X_valid, Y_valid, weights)
  print("Final loss on training: {}, final loss on validation: {}".format(loss1, loss2))

  print()
  print(Y_valid.reshape(-1))
  out = model.analyzer.modelA(X_valid)
  print(out.reshape(-1))

def analyzer_performance():
  # Get N random samples from pregenerated data.
  N = 50
  #with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    data_X = np.load(f)
    data_Y = np.load(f)
  """
  idx = np.random.choice(data_X.shape[0], N)
  X_valid = torch.from_numpy(data_X[idx, :]).float().to(model.device)
  Y_valid = torch.from_numpy(data_Y[idx, :]).float().to(model.device)
  """
  X_valid = torch.from_numpy(data_X[:N]).float().to(model.device)
  Y_valid = torch.from_numpy(data_Y[:N]).float().to(model.device)
  del data_X
  del data_Y

  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  threshold = 0.7
  for n in range(session.random_init + 1, session.N_tests):
    model.load(zeros("model_snapshot_", n), session.session_directory)
    idx1 = (Y_valid < threshold).reshape(-1)
    set_Y = Y_valid[idx1,:]
    set_X = X_valid[idx1,:]
    weights = model.analyzer.weights(set_Y)
    loss1 = model.analyzer.analyzer_loss(set_X, set_Y, weights)

    idx2 = (Y_valid >= threshold).reshape(-1)
    set_Y = Y_valid[idx2,:]
    set_X = X_valid[idx2,:]
    weights = model.analyzer.weights(set_Y)
    loss2 = model.analyzer.analyzer_loss(set_X, set_Y, weights)
    print("{}: {}, {}; {} {}".format(n, loss1, loss2, sum(idx1), sum(idx2)))

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("The command line arguments should specify sut_id, model_id, and a session directory.")
    raise SystemExit

  sut_id = sys.argv[1]
  if not sut_id in config["available_sut"]:
    raise ValueError("The sut_id '{}' is invalid.".format(sut_id))

  model_id = sys.argv[2]
  if not model_id in config["available_model"]:
    raise ValueError("The model_id '{}' is invalid.".format(model_id))

  sessions = []
  if sys.argv[3] == "-d":
    if len(sys.argv) < 5:
      raise ValueError("A directory must be specified after -d.")
    for directory in os.listdir(sys.argv[4]):
      if not directory.startswith("2021"): continue
      session = Session(model_id, sut_id, directory)
      session.add_saved_parameter(*config["session_attributes"][model_id])
      session.session_directory = os.path.join(sys.argv[4], directory)
      session.load()
      sessions.append(session)
  else:
    session_id = os.path.basename(os.path.normpath(sys.argv[3]))
    session = Session(model_id, sut_id, session_id)
    session.add_saved_parameter(*config["session_attributes"][model_id])
    session.session_directory = sys.argv[3]
    session.load()
    sessions.append(session)

  model, _view_test, _save_test = get_model(sut_id, model_id)

  view_test = lambda t: _view_test(t)
  save_test = lambda t, f: _save_test(t, session, f)

  for session in sessions:
    evaluate_session(session)
