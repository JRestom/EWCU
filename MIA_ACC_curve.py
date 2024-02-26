

import matplotlib.pyplot as plt

# # acc_train on Cifar
# data = {
#     'Ours': {'MIA': [0.51, 0.52, 0.55, 0.57], 'Accuracy': [98.3, 99.3, 99.7, 98 ]},
#     'Scrub': {'MIA': [0.524, 0.54, 0.554, 0.57 ], 'Accuracy': [98.7, 99.3, 99.8, 99.9]},
#     'CF-K ': {'MIA': [0.513, 0.547, 0.57, ], 'Accuracy': [93.8, 98.4, 97.9]},
#     'CF-K*': {'MIA': [0.523, 0.566, 0.573 ], 'Accuracy': [99, 98.7, 98 ]},
#     'Finetuning': {'MIA': [0.507], 'Accuracy': [96.8]},
#     'Bad-T': {'MIA': [0.522], 'Accuracy': [87.5]},
#     'Neg grad': {'MIA': [0.55], 'Accuracy': [93.2]},
#     'Adv Neg grad': {'MIA': [0.563], 'Accuracy': [97.5]},

# }

# # acc_test on Cifar
# data = {
#     'Ours': {'MIA': [0.51, 0.52, 0.55, 0.57], 'Accuracy': [84.2, 86.3, 87.4, 86.5 ]},
#     'Scrub': {'MIA': [0.524, 0.54, 0.554, 0.57 ], 'Accuracy': [86.8, 87.3, 87.8, 88.2]},
#     'CF-K ': {'MIA': [0.513, 0.547, 0.57, ], 'Accuracy': [82.7, 86.2, 86.5]},
#     'CF-K*': {'MIA': [0.523, 0.566, 0.573 ], 'Accuracy': [86.8, 86.9, 86.5]},
#     'Finetuning': {'MIA': [0.507], 'Accuracy': [83.1]},
#     'Bad-T': {'MIA': [0.522], 'Accuracy': [76.5]},
#     'Neg grad': {'MIA': [0.55], 'Accuracy': [82.5]},
#     'Adv Neg grad': {'MIA': [0.563], 'Accuracy': [86.5]},

# }

# acc_test on MUFAC
data = {
    'Ours': {'MIA': [0.63, 0.64, 0.68, 0.70], 'Accuracy': [60, 60, 58, 59 ]},
    'Scrub': {'MIA': [0.62, 0.71], 'Accuracy': [57.6, 59]},
    'CF-K ': {'MIA': [0.715], 'Accuracy': [58]},
    'Finetuning': {'MIA': [0.71], 'Accuracy': [60]},
    'Bad-T': {'MIA': [0.58], 'Accuracy': [24]},
    'Neg grad': {'MIA': [0.55], 'Accuracy': [40]},
    'Adv Neg grad': {'MIA': [0.60], 'Accuracy': [56]},

}


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  

plt.figure(figsize=(10, 6))

for (method, data), color in zip(data.items(), colors):
    mia_scores_desc = data["MIA"][::-1]  # Reverse the MIA scores for descending order
    accuracies_desc = data["Accuracy"][::-1]  # Reverse accuracies to match MIA score order
    plt.plot(mia_scores_desc, accuracies_desc, '-o', label=method, color=color)

plt.xlabel('MIA Score')
plt.ylabel('Accuracy')
plt.title('MIA Score vs. Test Accuracy')
plt.legend()
# Invert the x-axis to display MIA scores in descending order
plt.gca().invert_xaxis()
plt.savefig('MIA_test_mufac.pdf')
plt.show()


