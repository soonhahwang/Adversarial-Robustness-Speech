import validate
import fgsm
import train
from validate import validate_single_audio
import matplotlib.pyplot as plt
import numpy as np

vanilla_accuracy = np.load('vanilla_accuracy.npy')

print("Vanilla : ", vanilla_accuracy)
accuracy = []
accuracy_fgsm = []
SNRs = [5, 10, 15, 20, 25, 30, 35, 40, 45]
iteration_snr = []
iteration_snr_advtrain = []
accuracy = []
accuracy_advtrain = []
idx = 0

# FGSM_acc = []
# PGD_acc = []

# for perturb_snr in [25, 30, 35, 40]:
#     model = f'saved_model/dscnn_advtrain_fgsm_snr{perturb_snr}.h5'
#     for snr in SNRs:
#         FGSM_acc.append([perturb_snr, snr, validate.validate('fgsm', snr, False, model=model)])
#         for iteration in SNRs:
#             PGD_acc.append([perturb_snr, snr, iteration, validate.validate('pgd', snr, False, iteration, model)])

# np.save("FGSM_Attack_acc.npy", FGSM_acc)
# np.save("PGD_Attack_acc.npy", PGD_acc)

# FGSM_acc = np.load("FGSM_Attack_acc.npy")
# PGD_acc = np.load("PGD_Attack_acc.npy")

# np.savetxt("PGD_acc.csv", PGD_acc, delimiter=',')

# print(FGSM_acc)
# print(PGD_acc)

for perturb_snr in [25, 30, 35, 40]:
    model = f'saved_model/dscnn_advtrain_fgsm_snr{perturb_snr}.h5'
    accuracy = validate.validate(type='raw', model=model)
    print(perturb_snr, accuracy)
    
(5,	6.11E-01)
(10,	7.47E-01)
(15,	8.08E-01)
(20,	8.54E-01)
(25,	8.31E-01)
(30,	8.92E-01)
(35,	8.86E-01)
(40,	9.11E-01)
(45,	9.14E-01)

